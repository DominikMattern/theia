from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from collections.abc import Iterable, Iterator
from typing import Final, NamedTuple
from warnings import warn
from ctypes import Structure, addressof, c_float, c_uint32, memmove, sizeof
from hephaistos.glsl import vec3, mat
from scipy.integrate import cumulative_trapezoid
from scipy.special import ndtri, pdtrik
from theia.compiler import createPreamble, loadShader

from theia.light import LightSource, WavelengthSource, frankTamm, _make_scalar_array
from theia.lookup import Table
from theia.material import Medium, MaterialStore
from theia.property import TableProperty
import theia.units as u

__all__ = [
    "EnergyDepositionEvents",
    "ScintillationCherenkovDepositionEventSource",
    "ScintillationDepositionEventSource",
    "buildCherenkovTables",
    "createCombinedTasksFromEvents",
    "createTasksFromEvents",
    "iterCombinedTasksFromEvents",
    "iterTasksFromEvents",
]


def __dir__():
    return __all__


@dataclass
class EnergyDepositionEvents:
    """
    Data class storing energy deposition events.

    Every per-step quantity is stored as a NumPy array of equal length, with the
    k-th entry across all arrays describing the k-th deposition step. The start
    and end positions are split into separate arrays for each coordinate.
    """
    eventId: np.ndarray
    groupId: np.ndarray
    medium: np.ndarray
    startPositionX: np.ndarray
    startPositionY: np.ndarray
    startPositionZ: np.ndarray
    endPositionX: np.ndarray
    endPositionY: np.ndarray
    endPositionZ: np.ndarray
    startTime: np.ndarray
    startVelocity: np.ndarray
    endVelocity: np.ndarray
    energyDeposition: np.ndarray
    particleCode: np.ndarray

    def __len__(self) -> int:
        return len(self.eventId)


class _ScintillationComponent(NamedTuple):
    """One scintillation component of a (medium, particle code) pair."""

    component: int          #: 1, 2 or 3 - the suffix of its property names
    effectiveYield: float   #: photons per deposited energy, branching folded in
    timeConstant: float     #: decay time of the exponential emission
    spectrum: int           #: GPU address of the component's ppf table
    resolutionScale: float  #: widens the Gaussian around the photon number


def _scintillation_components(medium: Medium, particleCode: int):
    """
    The scintillation components one particle type excites in `medium`.

    Returns a list of `_ScintillationComponent`, one per active component, empty
    if the particle does not scintillate here. `_iter_scintillation_tasks` turns
    each of them into a segment of its own, so the properties travel with the
    segment instead of selecting which batch it lands in.

    Components are taken in order until their branching ratios add up to one. 
    Running out of components before reaching one warns and drops the remaining
    light.

    Property names
    --------------
    Properties are named ``scintillation_{key}_{component}_{particleCode}``, where
    the particle code is the *absolute* PDG code (so ``_11`` covers electrons and
    positrons) and both suffixes are optional. They are looked up from the most
    specific name to the least, dropping the particle code before the component:

    ==========================================  ==============================
    ``scintillation_{key}_{component}_{code}``   this component, this particle
    ``scintillation_{key}_{component}_ion``      this component, any nucleus
    ``scintillation_{key}_{component}``          this component, any particle
    ``scintillation_{key}_{code}``               any component, this particle
    ``scintillation_{key}_ion``                  any component, any nucleus
    ``scintillation_{key}``                      any component, any particle
    ==========================================  ==============================

    The ``_ion`` entries are only consulted for nuclear codes
    (``code > 1000000000``, i.e. alphas, deuterons and heavy nuclei), which share
    one fallback.

    Two keys are special. ``yield`` takes no component: it is the total number of
    photons per deposited energy, which the branching ratios then split up.
    ``branching_ratio`` is the opposite and *requires* one, since it is by
    definition the share of one particular component.
    """
    code = abs(int(particleCode))
    COMPONENT_REQUIRED = {"scintillation_branching_ratio"}

    def prop(key: str, component: int | None = None):
        """The most specific property defined for `key`, or `None`."""
        stems = []
        if component is not None:
            stems.append(f"scintillation_{key}_{component}")
        if key not in COMPONENT_REQUIRED:
            stems.append(f"scintillation_{key}")
        for stem in stems:
            candidates = [f"{stem}_{code}"]
            if code > 1000000000:
                candidates.append(f"{stem}_ion")
            candidates.append(stem)
            for candidate in candidates:
                if candidate in medium.properties:
                    return medium.properties[candidate]
        return None

    photonYield = prop("yield")
    if photonYield is None:
        return []

    components: list = []
    branching = 0.0
    for component in (1, 2, 3):
        bR = prop("branching_ratio", component)
        tC = prop("time_constant", component)
        spectrum = prop("spectrum_sampler", component)
        if bR is None or tC is None or spectrum is None:
            if components:
                warn(f"Medium '{medium.name}': branching ratio < 1 for particle "
                     f"{particleCode} but no scintillation component {component} "
                     "defined; ignoring the remaining light.")
            return components
        rs = prop("resolutionscale", component)
        components.append(_ScintillationComponent(
            component=component,
            effectiveYield=photonYield.value[1] * bR.value[1],
            timeConstant=tC.value[1],
            spectrum=spectrum.data,
            resolutionScale=rs.value[1] if rs is not None else 1.0,
        ))
        branching += bR.value[1]
        if 1.0 - branching <= 1e-5:
            break

    return components


def _cherenkov_charges(
    particleCode, charges: dict[int, float] | None = None
) -> np.ndarray:
    """
    Electric charge magnitude |z| in units of e, per entry of `particleCode`.

    A code listed in `charges` is given that charge. Anything not listed falls back 
    on the PDG convention: `DEFAULT_CHERENKOV_CHARGES` for the named particles and
    the PDG ion charge for ions. Careful with that ion charge: Z out of the code is 
    the charge of a **fully stripped** nucleus, not its actual ionisation state.
    """
    
    #: Electric charge in units of e, by absolute PDG code. A particle missing here
    #: and not recognised as a nucleus is taken to be neutral and radiates nothing.
    DEFAULT_CHERENKOV_CHARGES: Final[dict[int, float]] = {
        11: 1.0,     # e
        13: 1.0,     # mu
        15: 1.0,     # tau
        211: 1.0,    # pi
        321: 1.0,    # K
        2212: 1.0,   # p
        3222: 1.0,   # Sigma+
        3112: 1.0,   # Sigma-
        3312: 1.0,   # Xi-
        3334: 1.0,   # Omega-
    }
    #: Lowest nuclear code in the PDG scheme, which reads 10LZZZAAAI.
    PDG_ION_CODE: Final[int] = 1000010000

    code = np.abs(np.asarray(particleCode, np.int64))
    # PDG ion charge
    z = np.where(code >= PDG_ION_CODE, (code // 10000) % 1000, 0).astype(np.float64)
    for table in (DEFAULT_CHERENKOV_CHARGES, charges or {}):
        # one pass per listed code; `code` holds the handful of distinct
        # particles a chunk contains, not its steps
        for c, q in table.items():
            z[code == abs(int(c))] = abs(float(q))
    return z


#: Medium property holding the 2D Cherenkov percent point function, whose slow
#: axis is beta and whose fast axis is the quantile.
CHERENKOV_SPECTRUM_PROPERTY: Final[str] = "cherenkov_spectrum_sampler"
#: Medium property holding the Frank-Tamm photon count per unit length, sampled
#: over beta from the medium's threshold up to one.
CHERENKOV_YIELD_PROPERTY: Final[str] = "cherenkov_yield"


def buildCherenkovTables(
    media: Iterable[Medium], *, betaNodes: int = 64, tableValues: int = 1024,
    yieldSamples: int = 1024,
) -> list[Medium]:
    """
    Derives the Cherenkov emission of every medium from its refractive index and
    attaches it as two medium properties.

    Must be called **before** the media are handed to a `MaterialStore`: the
    store uploads every table property in one go and hands out its device
    address, so a table attached afterwards is never written.

    A medium with no refractive index, or none above one, cannot radiate and is
    left untouched. Calling this twice on the same medium simply recomputes the
    tables.

    Parameters
    ----------
    media: Iterable[Medium]
        Media to derive the Cherenkov emission for. Modified in place.
    betaNodes: int, default=64
        Rows of the emission table, i.e. how finely the spectrum resolves beta.
        The photon *number* is always taken at the step's exact beta, so this
        only affects the spectral shape; 64 keeps it well inside a per mille.
    tableValues: int, default=1024
        Columns of the emission table, i.e. how finely each row resolves the
        quantile of the emission spectrum.
    yieldSamples: int, default=1024
        Nodes of the yield curve, i.e. how finely the photon number per unit
        length resolves beta.

    Returns
    -------
    radiating: list[Medium]
        The media that were given tables, in input order.

    See Also
    --------
    theia.scintillation.createCombinedTasksFromEvents
    theia.scintillation.iterCombinedTasksFromEvents
    """
    radiating = []
    for medium in media:
        tables = _integrate_cherenkov_emission(
            medium, betaNodes=betaNodes, tableValues=tableValues,
            yieldSamples=yieldSamples)
        if tables is None:
            continue
        ppf, yieldGrid, betaMin = tables
        medium.properties[CHERENKOV_SPECTRUM_PROPERTY] = TableProperty(
            Table(ppf, (betaMin, 1.0), (0.0, 1.0)))
        medium.properties[CHERENKOV_YIELD_PROPERTY] = TableProperty(
            Table(yieldGrid, (betaMin, 1.0)))
        radiating.append(medium)
    return radiating


def _integrate_cherenkov_emission(
    medium: Medium, *, betaNodes: int, tableValues: int, yieldSamples: int,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """
    Integrates the Cherenkov emission of a medium over its refractive index.

    Returns ``(ppf, yieldGrid, betaMin)``, or `None` if the medium has no
    refractive index, or none above one, in which case it cannot radiate.

    Yield and spectrum are integrated over the range of the refractive index
    table itself.
    """
    prop = medium.properties.get("refractive_index")
    table = getattr(prop, "table", None)
    if table is None:
        return None
    n = np.asarray(table.samples, np.float64).ravel()
    lam = np.linspace(*table.axisRanges[0], len(n))
    nMax = float(n.max())
    if not nMax > 1.0:
        return None
    betaMin = 1.0 / nMax

    def density(beta):
        """Frank-Tamm d2N/(dx dlam) in [1/m 1/nm], clipped at the threshold."""
        b = np.atleast_1d(np.asarray(beta, np.float64))[:, None]
        return np.clip(frankTamm(lam, n[None, :], beta=b), 0.0, None)

    # yield curve, one np.interp per segment later on
    betaGrid = np.linspace(betaMin, 1.0, yieldSamples)
    yieldGrid = np.trapezoid(density(betaGrid), lam, axis=-1)

    # 2D percent point function, rows are beta, columns the quantile.
    # The bottom row sits exactly at the threshold and is degenerate (all
    # photons at lam_min).
    nodes = np.linspace(betaMin, 1.0, betaNodes)
    dens = density(nodes)
    # `initial=0` supplies the value at lam[0] - the integral over an empty
    # interval - so the result lines up index for index with `lam`, which is
    # what inverting it below needs. Same idiom as `media._compute_ppf`.
    cdf = cumulative_trapezoid(dens, lam, axis=-1, initial=0.0)
    total = cdf[:, -1:]
    cdf = np.divide(cdf, total, out=np.zeros_like(cdf), where=total > 0.0)
    quantile = np.linspace(0.0, 1.0, tableValues)
    ppf = np.stack([
        np.interp(quantile, cdf[i], lam) if total[i, 0] > 0.0
        else np.full(tableValues, lam[0])
        for i in range(betaNodes)
    ])
    return ppf, yieldGrid, betaMin


@dataclass(frozen=True)
class _CherenkovEmission:
    """
    Everything the producer needs to turn a step into a Cherenkov segment.

    `yieldGrid` samples the Frank-Tamm photon count per unit length over
    `betaGrid`, so a segment's yield is a single `np.interp`. `tableAddress`
    points at the 2D percent point function `buildCherenkovTables` put into the
    medium properties, i.e. the emission spectrum interpolates in beta rather
    than being binned.
    """

    betaGrid: np.ndarray
    yieldGrid: np.ndarray
    tableAddress: int
    betaMin: float
    nMax: float


def _read_cherenkov_emission(medium: Medium) -> _CherenkovEmission | None:
    """
    Reads back the Cherenkov tables `buildCherenkovTables` attached to a medium.

    Returns `None` if the medium carries no tables, i.e. cannot radiate. Raises
    if the tables are there but were never uploaded, which is what a medium that
    went into its `MaterialStore` before `buildCherenkovTables` looks like.
    """
    sampler = medium.properties.get(CHERENKOV_SPECTRUM_PROPERTY)
    yieldProp = medium.properties.get(CHERENKOV_YIELD_PROPERTY)
    if sampler is None or yieldProp is None or yieldProp.table is None:
        return None
    if not sampler.data:
        raise ValueError(
            f"medium '{medium.name}' carries Cherenkov tables whose address is "
            "zero, i.e. they were never uploaded; call 'buildCherenkovTables' "
            "on the media *before* handing them to the MaterialStore")
    # The beta axis of the yield curve is where the threshold - and with it the
    # largest refractive index - comes from.
    betaMin, betaMax = yieldProp.table.axisRanges[0]
    yieldGrid = np.asarray(yieldProp.table.samples, np.float64)
    betaGrid = np.linspace(betaMin, betaMax, len(yieldGrid))
    return _CherenkovEmission(betaGrid, yieldGrid, int(sampler.data),
                              float(betaMin), 1.0 / float(betaMin))


@dataclass
class _Segments:
    """
    One batch element per entry: the light a step emits into one component.

    Purely a bundle of parallel columns - the generic helpers below iterate the
    fields, so adding a column does not mean touching the concatenation, the
    slicing and the padding on top of the producer.
    """

    startPosition: np.ndarray        # (n, 3) float32
    endPosition: np.ndarray          # (n, 3) float32
    mediumIdx: np.ndarray            # uint32, slot in the MaterialStore
    startTime: np.ndarray            # float32
    startVelocity: np.ndarray        # float32
    endVelocity: np.ndarray          # float32
    timeConstant: np.ndarray         # float32, zero for Cherenkov
    expectedPhotonNumber: np.ndarray # float64, the mean this segment emits
    spectrum: np.ndarray             # uint64, GPU address of the ppf
    eventId: np.ndarray              # int64
    groupId: np.ndarray              # int64
    axisValue: np.ndarray            # float32, NaN selects a 1D spectrum
    mode: np.ndarray                 # uint32, 0 scintillation / 1 Cherenkov
    pathCount: np.ndarray            # uint64, paths this segment owns

    #: Columns the light source consumes verbatim, as `lightSource__{name}`.
    _lightSourceNames = ("startPosition", "endPosition", "mediumIdx", "startTime",
                         "startVelocity", "endVelocity", "timeConstant",
                         "expectedPhotonNumber")
    _names = (*_lightSourceNames, "spectrum", "eventId", "groupId",
              "axisValue", "mode", "pathCount")

    def __len__(self) -> int:
        return len(self.startTime)

    def columns(self) -> dict:
        """The columns by name, for handing to a task."""
        return {name: getattr(self, name) for name in self._names}

    def __getitem__(self, key) -> "_Segments":
        """A view (or copy, depending on `key`) of the same columns."""
        return _Segments(**{n: getattr(self, n)[key] for n in self._names})

    def copy(self) -> "_Segments":
        return _Segments(**{n: getattr(self, n).copy() for n in self._names})

    def concat(self, other: "_Segments") -> "_Segments":
        return _Segments(**{n: np.concatenate((getattr(self, n), getattr(other, n)))
                            for n in self._names})

    def padded(self, total: int) -> "_Segments":
        """
        Grown to `total` entries with segments that emit nothing.

        Zero for the path count - so the shader's search can never land on them -
        and for the coordinates. Everything else repeats the last real value
        rather than zeroing: a zero event or group id would make the last task
        look like it ended on event 0 of group 0, a zero spectrum address would
        be a null table, and forcing `axisValue` to NaN could pair a 2D table
        with a 1D lookup.
        """
        pad = total - len(self)
        if pad <= 0:
            return self
        out = {}
        for name in self._names:
            column = getattr(self, name)
            if name in ("pathCount", "startPosition", "endPosition"):
                fill = np.zeros((pad, *column.shape[1:]), column.dtype)
            else:
                fill = np.repeat(column[-1:], pad, axis=0)
            out[name] = np.concatenate((column, fill))
        return _Segments(**out)


@dataclass(frozen=True)
class _PairIndex:
    """
    The (medium, particle code) pairs a chunk contains, one entry per pair.

    What a step emits follows from its pair alone, so every emission property is
    resolved once per pair and a step only ever carries an index into these.
    """

    pairOfStep: np.ndarray      #: intp, the pair of every step
    media: list                 #: `Medium` per pair
    codes: list                 #: PDG particle code per pair
    mediumIdx: np.ndarray       #: uint32, slot in the MaterialStore per pair

    def __len__(self) -> int:
        return len(self.codes)


def _group_steps_by_pair(chunk, matStore) -> _PairIndex:
    """Groups the deposition steps of a chunk by their (medium, code) pair."""
    numSteps = len(chunk.eventId)
    # compact index per medium, by identity: `Medium` is a dataclass, so `==`
    # would compare every field numSteps times over, and `np.unique` on the
    # objects would need an ordering they do not have
    _, firstStepOfMedium, mediumOfStep = np.unique(
        np.fromiter((id(m) for m in chunk.medium), np.int64, numSteps),
        return_index=True, return_inverse=True,
    )
    media = [chunk.medium[i] for i in firstStepOfMedium]
    storeIdxOfMedium = np.array([matStore.media[m.name] for m in media], np.uint32)

    codes, codeOfStep = np.unique(np.asarray(chunk.particleCode),
                                  return_inverse=True)
    # The pair of every step, as a compact index into the handful of pairs a
    # chunk actually contains. The key is linearized rather than handed to a two
    # column `np.unique(..., axis=0)`, whose lexsort would run over every step;
    # it is decoded here, once, and never again.
    pairKeys, pairOfStep = np.unique(mediumOfStep * len(codes) + codeOfStep,
                                     return_inverse=True)
    mediumOfPair, codeSlotOfPair = np.divmod(pairKeys, len(codes))
    return _PairIndex(
        pairOfStep=pairOfStep,
        media=[media[slot] for slot in mediumOfPair],
        codes=codes[codeSlotOfPair].tolist(),
        mediumIdx=storeIdxOfMedium[mediumOfPair],
    )


@dataclass(frozen=True)
class _PairProperties:
    """
    The emission properties of one pair, one entry per pair of a `_PairIndex`.
    """

    components: list        #: list of `_ScintillationComponent` per pair
    cherenkov: list         #: `_CherenkovEmission | None` per pair
    charge: np.ndarray      #: float64, |z| in units of e per pair

    @property
    def anyCherenkov(self) -> bool:
        """Whether any pair of the chunk can radiate at all."""
        return any(c is not None for c in self.cherenkov)


def _resolve_pair_properties(pairs: _PairIndex, scintillationCache: dict,
                             cherenkovCache: dict, *, cherenkov: bool,
                             cherenkovCharges) -> _PairProperties:
    """
    Reads the emission properties of every pair, memoizing them across chunks.

    A pair radiates Cherenkov light if its particle is charged and its medium
    carries the tables `buildCherenkovTables` attaches; everything that depends
    on the step itself (beta, length) is decided further down.
    """
    charge = (_cherenkov_charges(pairs.codes, cherenkovCharges) if cherenkov
              else np.zeros(len(pairs)))
    components: list = []
    cherenkovOfPair: list = [None] * len(pairs)
    for pair, (medium, code) in enumerate(zip(pairs.media, pairs.codes)):
        # keyed by name, not identity: an id() can be recycled once a Medium is
        # collected, and a name is how media are identified everywhere else
        cacheKey = (medium.name, code)
        if cacheKey not in scintillationCache:
            scintillationCache[cacheKey] = _scintillation_components(medium, code)
        components.append(scintillationCache[cacheKey])
        if charge[pair] <= 0.0:
            continue
        if medium.name not in cherenkovCache:
            cherenkovCache[medium.name] = _read_cherenkov_emission(medium)
            if cherenkovCache[medium.name] is None:
                _warn_missing_cherenkov_tables(medium)
        cherenkovOfPair[pair] = cherenkovCache[medium.name]
    return _PairProperties(components, cherenkovOfPair, charge)


def _warn_missing_cherenkov_tables(medium: Medium) -> None:
    """
    Warns about a medium that could radiate but was never given its tables.
    """
    prop = medium.properties.get("refractive_index")
    table = getattr(prop, "table", None)
    if table is None or not np.asarray(table.samples).max() > 1.0:
        return
    warn(f"Medium '{medium.name}': refractive index above one, but no Cherenkov "
         "tables; run 'buildCherenkovTables' on the media before handing them to "
         "the MaterialStore or their Cherenkov light is dropped.")


@dataclass(frozen=True)
class _SlotTables:
    """
    The emission properties as one row per pair, padded to a common width.

    A segment picks its properties with a single (pair, slot) gather. The slots
    of a pair are its scintillation components in order, followed - where the
    pair can radiate - by one slot carrying its Cherenkov light.
    """

    photonYield: np.ndarray       #: float64, photons per deposited energy
    timeConstant: np.ndarray      #: float32, zero for Cherenkov
    spectrum: np.ndarray          #: uint64, GPU address of the ppf
    resolutionScale: np.ndarray   #: float64
    numComponents: np.ndarray     #: intp per pair, i.e. its Cherenkov slot


def _build_slot_tables(properties: _PairProperties) -> _SlotTables | None:
    """
    Lays the emission properties of every pair out as gatherable rows.

    Returns `None` if no pair of the chunk emits anything at all.
    """
    numPairs = len(properties.components)
    maxComponents = max((len(c) for c in properties.components), default=0)
    if maxComponents == 0 and not properties.anyCherenkov:
        return None
    # one extra slot holds the Cherenkov component
    numSlots = maxComponents + (1 if properties.anyCherenkov else 0)

    numComponents = np.array([len(c) for c in properties.components], np.intp)
    photonYield = np.zeros((numPairs, numSlots))
    timeConstant = np.zeros((numPairs, numSlots), np.float32)
    spectrum = np.zeros((numPairs, numSlots), np.uint64)
    resolutionScale = np.ones((numPairs, numSlots))
    for pair, components in enumerate(properties.components):
        for slot, component in enumerate(components):
            photonYield[pair, slot] = component.effectiveYield
            timeConstant[pair, slot] = component.timeConstant
            spectrum[pair, slot] = component.spectrum
            resolutionScale[pair, slot] = component.resolutionScale
        # the padded slots are never dereferenced - the shader's search cannot
        # land on a segment owning no path - but a null table address would be a
        # device fault if it ever were, so point them at a real table
        if components:
            spectrum[pair, len(components):] = components[-1].spectrum
    # The Cherenkov slot sits one past the pair's scintillation components. Its
    # yield is 1 because the photon number is written into
    # `expectedPhotonNumber` directly, and its time constant is zero because
    # Cherenkov emission is prompt.
    for pair, cherenkov in enumerate(properties.cherenkov):
        if cherenkov is None:
            continue
        slot = numComponents[pair]
        photonYield[pair, slot] = 1.0
        timeConstant[pair, slot] = 0.0
        spectrum[pair, slot] = cherenkov.tableAddress
        resolutionScale[pair, slot] = 1.0
    return _SlotTables(photonYield, timeConstant, spectrum, resolutionScale,
                       numComponents)


@dataclass(frozen=True)
class _RadiatingSteps:
    """Which steps radiate Cherenkov light, plus what decides how much."""

    radiates: np.ndarray            #: bool per step
    count: int                      #: radiating steps, i.e. Cherenkov segments
    betaBar: np.ndarray | None      #: mean beta per step, `None` if none radiate
    stepLength: np.ndarray | None   #: track length per step, `None` likewise


def _find_cherenkov_radiating_steps(
    chunk, pairs: _PairIndex, properties: _PairProperties,
) -> _RadiatingSteps:
    """
    Decides which deposition steps emit Cherenkov light.

    A step needs a charged particle in a medium with an index above one (already
    folded into `properties`), a beta above that medium's threshold, and a non
    zero length - the latter also keeps the shader from normalizing a zero
    direction.
    """
    numSteps = len(chunk.eventId)
    if not properties.anyCherenkov:
        return _RadiatingSteps(np.zeros(numSteps, bool), 0, None, None)
    betaBar = 0.5 * (np.asarray(chunk.startVelocity, np.float64)
                     + np.asarray(chunk.endVelocity, np.float64)) / u.c
    delta = [np.asarray(end, np.float64) - np.asarray(start, np.float64)
             for start, end in ((chunk.startPositionX, chunk.endPositionX),
                                (chunk.startPositionY, chunk.endPositionY),
                                (chunk.startPositionZ, chunk.endPositionZ))]
    stepLength = np.sqrt(sum(d * d for d in delta))
    nMaxOfPair = np.array(
        [0.0 if c is None else c.nMax for c in properties.cherenkov], np.float64)
    radiates = (betaBar * nMaxOfPair[pairs.pairOfStep] > 1.0) & (stepLength > 0.0)
    # every radiating step gains exactly one segment, so this is also the number
    # of Cherenkov segments - no reduction over the segments needed
    return _RadiatingSteps(radiates, int(radiates.sum()), betaBar, stepLength)


def _expand_steps_to_segments(numSegmentsOfStep: np.ndarray):
    """
    Turns a per step segment count into the (step, slot) index of every segment.

    Returns ``(stepOfSegment, slotOfSegment)``. A segment's slot is its index
    counted from the start of its step's run, which is what makes the pair's
    property row gatherable.
    """
    numSteps = len(numSegmentsOfStep)
    # where every step's run of segments begins, so the last entry is the total
    segmentStart = np.zeros(numSteps + 1, np.intp)
    np.cumsum(numSegmentsOfStep, out=segmentStart[1:])
    numSegments = int(segmentStart[-1])
    stepOfSegment = np.repeat(np.arange(numSteps), numSegmentsOfStep)
    slotOfSegment = np.arange(numSegments) - segmentStart[stepOfSegment]
    return stepOfSegment, slotOfSegment


def _compute_cherenkov_photons(steps: np.ndarray, pairs: _PairIndex,
                               properties: _PairProperties,
                               radiating: _RadiatingSteps) -> np.ndarray:
    """
    The photon number the Cherenkov segment of every step in `steps` emits.

    One `np.interp` against the medium's yield curve per pair that actually
    radiates, times the step length and the charge squared. Working on the
    Cherenkov segments rather than on all steps keeps every mask here as short
    as the light there is.
    """
    pairOfSegment = pairs.pairOfStep[steps]
    photons = np.empty(len(steps))
    for pair in np.unique(pairOfSegment):
        cherenkov = properties.cherenkov[pair]
        mask = pairOfSegment == pair
        radiatingSteps = steps[mask]
        photons[mask] = (
            np.interp(radiating.betaBar[radiatingSteps], cherenkov.betaGrid,
                      cherenkov.yieldGrid)
            * radiating.stepLength[radiatingSteps] * properties.charge[pair] ** 2)
    return photons


#: Expected photon number up to which a scintillation segment draws its photon
#: number from a Poisson rather than a Gaussian.
SCINTILLATION_POISSON_THRESHOLD: Final[float] = 10.0


def _draw_scintillation_counts(mean: np.ndarray, resolutionScale: np.ndarray,
                               uniform: np.ndarray) -> np.ndarray:
    """
    The scintillation photon numbers, one per entry of `uniform`.

    Gaussian with `resolutionScale` widening it, but - like Geant4 - Poisson
    below `SCINTILLATION_POISSON_THRESHOLD`: there the Gaussian would round
    small means away or reach below zero, and clipping it biases the photon
    number up. Both invert the same uniform, so it stays one draw per segment.
    """
    count = np.rint(mean + np.sqrt(mean) * resolutionScale * ndtri(uniform))
    small = mean <= SCINTILLATION_POISSON_THRESHOLD
    # see the Cherenkov draw for why `pdtrik` is rounded up
    count[small] = np.ceil(pdtrik(uniform[small], mean[small]))
    return count


def _draw_photon_counts(expectedPhotons: np.ndarray, slots: _SlotTables,
                        pairOfSegment: np.ndarray, slotOfSegment: np.ndarray,
                        isCherenkov: np.ndarray, numCherenkov: int,
                        rng, cherenkovRng) -> np.ndarray:
    """
    Draws the photon number of every segment around its expected value.

    Gaussian for scintillation, to incorporate the resolution scale, switching to
    Poisson for small means, and Poisson for Cherenkov. One draw per segment,
    taken here rather than per task so that segment k always gets draw k of the
    run: `random` consumes exactly one 64 bit word per value, so the stream is
    position addressable and the photon numbers
    do not depend on how `events` is chunked nor on `eventsPerBatch`. The
    scintillation stream is consumed only by scintillation segments, so
    scintillation segment k still gets draw k whether or not Cherenkov is
    switched on - an existing run stays bit for bit reproducible.
    """
    numSegments = len(expectedPhotons)
    # `ndtri` is the exact inverse normal CDF
    uniform = np.maximum(rng.random(numSegments - numCherenkov), 5e-324)
    if numCherenkov == 0:
        # every segment is a scintillation one, so the draws line up with the
        # segments and none of the masking below is needed
        resolutionScale = slots.resolutionScale[pairOfSegment, slotOfSegment]
        return _draw_scintillation_counts(expectedPhotons, resolutionScale,
                                          uniform)
    pathCount = np.empty(numSegments)
    isScint = ~isCherenkov
    resolutionScale = slots.resolutionScale[pairOfSegment[isScint],
                                            slotOfSegment[isScint]]
    pathCount[isScint] = _draw_scintillation_counts(
        expectedPhotons[isScint], resolutionScale, uniform)
    # `pdtrik` inverts the Poisson CDF *continuously*, so it has to be rounded
    # up to land on the discrete quantile min{k : P(K <= k) >= u}.
    uCherenkov = np.maximum(cherenkovRng.random(numCherenkov), 5e-324)
    pathCount[isCherenkov] = np.ceil(
        pdtrik(uCherenkov, expectedPhotons[isCherenkov]))
    return pathCount


def _count_rays(expectedPhotons: np.ndarray, raysPerPhoton: float) -> np.ndarray:
    """
    A fixed number of rays per segment, so that the segment - and with it the
    event - follows from the path index.
    """
    return np.where(expectedPhotons > 0.0,
                    np.maximum(1.0, np.rint(expectedPhotons * raysPerPhoton)),
                    0.0)


def _expand_segments(chunk, matStore, scintillationCache, cherenkovCache, rng, *,
                     emitParticles, raysPerPhoton,
                     cherenkov=False, cherenkovRng=None, cherenkovCharges=None):
    """
    Turns one input chunk into a stream of segments, in input order.

    A deposition step excites up to three scintillation components, each emitting
    with its own yield, time constant and spectrum. They become *consecutive*
    segments of the stream, so all segments of a step - and hence of an event -
    stay adjacent.

    With `cherenkov`, a step may gain one further segment carrying its Cherenkov
    light. It is treated as just another component so that it lands next to the
    step's scintillation segments, but unlike those its presence is decided per
    step rather than per pair: only a charged particle above the medium's
    threshold radiates.

    Everything is whole array work. The component set of a step follows from its
    (medium, particle code) pair alone, so the properties are resolved once per
    pair rather than once per step.

    Returns a `_Segments` holding all columns, or `None` if the chunk contributes
    no light at all.
    """

    # We will use the following convention for variable names:
    #   Step: One entry of EnergyDepositionEvents
    #   Segment: One entry of ScintillationDepositionEventSource's inputs
    #   Component: One of (up to 3) scintillation components of a given medium
    #   Pair: The combination (medium, particle code) that determines the scintillation
    #         properties
    #   Slot: A column of the per pair property tables - one per component, plus
    #         one for Cherenkov light where a pair can radiate
    # Each step will lead to [number of slots] segments.

    if len(chunk.eventId) == 0:
        return None

    pairs = _group_steps_by_pair(chunk, matStore)
    properties = _resolve_pair_properties(pairs, scintillationCache,
                                          cherenkovCache, cherenkov=cherenkov,
                                          cherenkovCharges=cherenkovCharges)
    slots = _build_slot_tables(properties)
    if slots is None:
        return None
    radiating = _find_cherenkov_radiating_steps(chunk, pairs, properties)

    # expand: one segment per (step, component), plus the Cherenkov one
    numSegmentsOfStep = slots.numComponents[pairs.pairOfStep] + radiating.radiates
    stepOfSegment, slotOfSegment = _expand_steps_to_segments(numSegmentsOfStep)
    numSegments = len(stepOfSegment)
    if numSegments == 0:
        return None
    pairOfSegment = pairs.pairOfStep[stepOfSegment]
    # a segment past its pair's scintillation components is the Cherenkov one
    isCherenkov = slotOfSegment >= slots.numComponents[pairOfSegment]

    expectedPhotons = (np.asarray(chunk.energyDeposition, np.float64)[stepOfSegment]
                       * slots.photonYield[pairOfSegment, slotOfSegment])
    # NaN tells the wavelength source to read a 1D table, i.e. leaves the
    # scintillation segments on the path they have always taken
    axisValue = np.full(numSegments, np.nan, np.float32)
    if radiating.count:
        cherenkovSteps = stepOfSegment[isCherenkov]
        expectedPhotons[isCherenkov] = _compute_cherenkov_photons(
            cherenkovSteps, pairs, properties, radiating)
        # beta on the second table axis. The shader recomputes it from the
        # velocities for the cone angle; both agree up to float32 rounding,
        # which the shader's clamp absorbs.
        axisValue[isCherenkov] = radiating.betaBar[cherenkovSteps]

    if emitParticles:
        pathCount = _draw_photon_counts(
            expectedPhotons, slots, pairOfSegment, slotOfSegment, isCherenkov,
            radiating.count, rng, cherenkovRng)
    else:
        pathCount = _count_rays(expectedPhotons, raysPerPhoton)

    def gatherPosition(x, y, z):
        # one pass, no np.stack().astype() intermediate
        out = np.empty((numSegments, 3), np.float32)
        out[:, 0] = np.asarray(x, np.float32)[stepOfSegment]
        out[:, 1] = np.asarray(y, np.float32)[stepOfSegment]
        out[:, 2] = np.asarray(z, np.float32)[stepOfSegment]
        return out

    return _Segments(
        startPosition=gatherPosition(chunk.startPositionX,
                                     chunk.startPositionY,
                                     chunk.startPositionZ),
        endPosition=gatherPosition(chunk.endPositionX,
                                   chunk.endPositionY,
                                   chunk.endPositionZ),
        mediumIdx=pairs.mediumIdx[pairOfSegment],
        startTime=np.asarray(chunk.startTime, np.float32)[stepOfSegment],
        startVelocity=np.asarray(chunk.startVelocity, np.float32)[stepOfSegment],
        endVelocity=np.asarray(chunk.endVelocity, np.float32)[stepOfSegment],
        timeConstant=slots.timeConstant[pairOfSegment, slotOfSegment],
        expectedPhotonNumber=expectedPhotons,
        spectrum=slots.spectrum[pairOfSegment, slotOfSegment],
        eventId=np.asarray(chunk.eventId, np.int64)[stepOfSegment],
        groupId=np.asarray(chunk.groupId, np.int64)[stepOfSegment],
        axisValue=axisValue,
        mode=isCherenkov.astype(np.uint32),
        pathCount=np.clip(pathCount, 0, None, out=pathCount).astype(np.uint64),
    )


def _iter_scintillation_tasks(events, matStore, *, eventsPerBatch, padIncomplete,
                              emitParticles, RNGSeed, raysPerPhoton=None,
                              withEventIds=False, cherenkov=False,
                              cherenkovCharges=None):
    """
    Core generator shared by `createTasksFromEvents` and `iterTasksFromEvents`.

    Emits tasks strictly in input order: the deposition steps of a chunk are
    expanded into segments (`_expand_segments`), appended to a single stream, and
    cut into batches of `eventsPerBatch`.

    A complete batch is a contiguous *view* into the expanded arrays; only the
    one batch straddling a chunk boundary is ever copied.
    """
    if emitParticles:
        if raysPerPhoton is not None:
            raise ValueError("'raysPerPhoton' is for rays; particles always get "
                             "one path per photon")
    else:
        if raysPerPhoton is None:
            raise ValueError(
                "rays need 'raysPerPhoton.")
        if not raysPerPhoton > 0.0:
            raise ValueError(
                f"'raysPerPhoton' must be positive, got {raysPerPhoton}")
        raysPerPhoton = float(raysPerPhoton)

    # One generator for the whole run, so segment k always gets draw k. Philox
    # and `random()` on purpose: `random()` consumes exactly one 64 bit word per
    # value, so the stream is position addressable and segment k's photon number
    # can be recomputed on its own. Philox emits four words per counter block, 
    # so the jump to draw k is `advance(k // 4)` plus `k % 4` discards. 
    # A missing seed is drawn once, here, rather than left to each Philox: the
    # two streams below have to be two substreams of *one* key, and a run is
    # only repeatable if its seed exists somewhere.
    if RNGSeed is None:
        RNGSeed = np.random.SeedSequence().entropy
    rng = np.random.Generator(np.random.Philox(key=RNGSeed))
    # Cherenkov draws from its own substream, so switching it on leaves the
    # scintillation photon numbers of an existing run unchanged.
    cherenkovRng = np.random.Generator(np.random.Philox(key=RNGSeed).jumped(1))

    scintillationCache: dict = {}    # (mediumName, particleCode) -> components
    cherenkovCache: dict = {}    # mediumName -> _CherenkovEmission | None
    pending: _Segments | None = None  # < eventsPerBatch, owning their memory

    def make_task(segments: _Segments):
        cumulativePaths = np.cumsum(segments.pathCount)
        numPaths = int(cumulativePaths[-1]) if len(cumulativePaths) else 0
        if numPaths > np.iinfo(np.uint32).max:
            raise ValueError(
                f"a batch would hold {numPaths} paths, more than the "
                f"{np.iinfo(np.uint32).max} the shader can index; lower "
                "eventsPerBatch or raysPerPhoton")
        task = {
            f"lightSource__{name}": getattr(segments, name)
            for name in _Segments._lightSourceNames
        }
        # the light source hands out this many paths per segment - photons for
        # particles, rays otherwise - and the wavelength source needs the same
        # numbers to find the same segment
        task["lightSource__numSamples"] = segments.pathCount
        task["photons__spectrumTableAddress"] = segments.spectrum
        task["photons__count"] = segments.pathCount
        if cherenkov:
            # which branch the light source takes, and where on the spectrum's
            # second axis this segment sits (NaN for scintillation)
            task["lightSource__mode"] = segments.mode
            task["photons__axisValue"] = segments.axisValue
        task["_batchSize"] = numPaths
        # (params, args): the scheduler hands `args` to the process callback, so a
        # response can attribute its hits back to events. `cumulativePaths` is the
        # path id boundary of every segment, `(groupId, eventId)` says which event
        # of which group owns it.
        return ((task, (segments.groupId, segments.eventId, cumulativePaths))
                if withEventIds else task)

    for chunk in events:
        segments = _expand_segments(chunk, matStore, scintillationCache,
                                    cherenkovCache, rng,
                                    emitParticles=emitParticles,
                                    raysPerPhoton=raysPerPhoton,
                                    cherenkov=cherenkov,
                                    cherenkovRng=cherenkovRng,
                                    cherenkovCharges=cherenkovCharges)
        if segments is None:
            continue
        numSegments = len(segments)
        begin = 0
        if pending is not None:
            # Fill up the carried over remainder and hand that one batch out;
            # only it is copied. Concatenating the whole chunk onto `pending`
            # instead would copy every column of it - hundreds of megabytes for a
            # large chunk - just to prepend a handful of segments.
            take = eventsPerBatch - len(pending)
            if numSegments < take:
                pending = pending.concat(segments)
                continue
            yield make_task(pending.concat(segments[:take]))
            pending, begin = None, take
        numComplete = begin + (numSegments - begin) // eventsPerBatch * eventsPerBatch
        for start in range(begin, numComplete, eventsPerBatch):
            yield make_task(segments[start:start + eventsPerBatch])
        # The remainder carries over to the next chunk and must own its memory:
        # the tasks handed out above are consumed asynchronously, so it may
        # neither alias nor pin their arrays.
        if numComplete < numSegments:
            pending = segments[numComplete:].copy()

    if pending is not None and padIncomplete:
        yield make_task(pending.padded(eventsPerBatch))


def createTasksFromEvents(events: Iterable[EnergyDepositionEvents],
                          matStore: MaterialStore,
                          *,
                          eventsPerBatch: int = 1,
                          padIncomplete: bool = True,
                          emitParticles: bool = False,
                          raysPerPhoton: float | None = None,
                          RNGSeed: int | None = None,
                          withEventIds: bool = False,
                          ) -> tuple[list, int | None]:
    """
    Utility function to prepare the simulation of scintillation light. Takes a
    list of events and returns a list of tasks which properly configures the light
    source and wavelength source for a scheduler, assuming a
    `ScintillationDepositionEventSource` and a `BatchedSpectrumWavelengthSource`
    are used. Scintillation spectrums, time constants, branching ratios and yields
    must be defined for all scintillating media through their medium properties
    (see `material.OpticalProperties` for the property names).

    Tasks are emitted **in input order**. All segments of a step, and hence of an
    event, are therefore adjacent.

    Parameters
    ----------
    events: Iterable[EnergyDepositionEvents]
        Collection of event containers to generate the tasks for.
    matStore: MaterialStore
        Material Store that is used for the simulation.
    eventsPerBatch: int, default=1
        Number of track segments per task. Note that this counts *segments*, i.e.
        one entry per scintillation component of a step, so a medium with two
        components needs twice the value for the same number of light paths per
        dispatch. The `ScintillationDepositionEventSource` and the
        `BatchedSpectrumWavelengthSource` must be created with the same number.
    padIncomplete: bool, default=True
        If `True`, the last batch is padded with zero-path segments when it holds
        fewer than `eventsPerBatch`. If `False` it is discarded. Unlike the old
        grouped batching this affects only *one* batch per run.
    emitParticles: bool, default=False
        Whether to emit particles instead of rays. If `True` this will draw a
        random number of photons per segment based on the expected value and the
        medium's resolution scale.
    raysPerPhoton: float | None, default=None
        Rays to sample per physical photon, required if `emitParticles` is
        `False`.
    RNGSeed: int | None, default=None
        The seed for the photon number RNG. Segment k always gets draw k of the
        run, so the photon numbers depend on neither the chunking of `events` nor
        `eventsPerBatch`.
    withEventIds: bool, default=False
        If `True`, every entry of `tasks` is ``(task, (groupId, eventId,
        cumulativePaths))`` instead of a bare task. `groupId` and `eventId` name
        the event each segment belongs to and `cumulativePaths` its path id
        boundary, which is what a response needs to attribute a hit back to an
        event.

    Returns
    -------
    tasks: list[dict] | list[tuple[dict, tuple]]
        List of tasks to give to the scheduler, paired with their event ids if
        `withEventIds` is `True`.
    maxBatchSize: int | None
        Largest number of light paths any of the tasks needs, or `None` if the
        tasks do not fix it. Use it as the tracer's capacity.

    See Also
    --------
    theia.scintillation.iterTasksFromEvents
    """
    tasks: list = []
    maxBatchSize = 0
    for item in _iter_scintillation_tasks(
        events,
        matStore,
        eventsPerBatch=eventsPerBatch,
        padIncomplete=padIncomplete,
        emitParticles=emitParticles,
        RNGSeed=RNGSeed,
        raysPerPhoton=raysPerPhoton,
        withEventIds=withEventIds,
    ):
        tasks.append(item)
        task = item[0] if withEventIds else item
        maxBatchSize = max(maxBatchSize, task["_batchSize"])

    return tasks, (None if maxBatchSize == 0 else maxBatchSize)


def iterTasksFromEvents(events: Iterable[EnergyDepositionEvents],
                        matStore: MaterialStore,
                        *,
                        eventsPerBatch: int = 1,
                        padIncomplete: bool = True,
                        emitParticles: bool = False,
                        raysPerPhoton: float | None = None,
                        RNGSeed: int | None = None,
                        withEventIds: bool = False,
                        ) -> Iterator[dict]:
    """
    Streaming variant of `createTasksFromEvents` yielding tasks lazily.

    Behaves exactly like `createTasksFromEvents` - identical order, batching and
    task layout - but returns a generator instead of a list. The segment stream is
    kept alive across the whole `events` iterable, so `events` may itself be a
    generator producing chunks (e.g. read from disk on demand) without changing
    the batching compared to processing everything at once.

    This enables overlapping event reading / task building with GPU work and keeps
    host memory bounded.

    Parameters
    ----------
    events: Iterable[EnergyDepositionEvents]
        Collection of event containers to generate the tasks for.
    matStore: MaterialStore
        Material Store that is used for the simulation.
    eventsPerBatch: int, default=1
        Number of track segments per task. Note that this counts *segments*, i.e.
        one entry per scintillation component of a step, so a medium with two
        components needs twice the value for the same number of light paths per
        dispatch. The `ScintillationDepositionEventSource` and the
        `BatchedSpectrumWavelengthSource` must be created with the same number.
    padIncomplete: bool, default=True
        If `True`, the last batch is padded with zero-path segments when it holds
        fewer than `eventsPerBatch`. If `False` it is discarded. Unlike the old
        grouped batching this affects only *one* batch per run.
    emitParticles: bool, default=False
        Whether to emit particles instead of rays. If `True` this will draw a
        random number of photons per segment based on the expected value and the
        medium's resolution scale.
    raysPerPhoton: float | None, default=None
        Rays to sample per physical photon, required if `emitParticles` is
        `False`.
    RNGSeed: int | None, default=None
        The seed for the photon number RNG. Segment k always gets draw k of the
        run, so the photon numbers depend on neither the chunking of `events` nor
        `eventsPerBatch`.
    withEventIds: bool, default=False
        If `True`, yield ``(task, (groupId, eventId, cumulativePaths))`` instead of
        a bare task. `groupId` and `eventId` name the event each segment belongs
        to and `cumulativePaths` its path id boundary, which is what a response
        needs to attribute a hit back to an event.

    Yields
    ------
    task: dict
        One task per completed batch, ready to hand to a scheduler.

    See Also
    --------
    theia.scintillation.createTasksFromEvents
    """
    yield from _iter_scintillation_tasks(
        events,
        matStore,
        eventsPerBatch=eventsPerBatch,
        padIncomplete=padIncomplete,
        emitParticles=emitParticles,
        RNGSeed=RNGSeed,
        raysPerPhoton=raysPerPhoton,
        withEventIds=withEventIds,
    )


def createCombinedTasksFromEvents(events: Iterable[EnergyDepositionEvents],
                                  matStore: MaterialStore,
                                  *,
                                  eventsPerBatch: int = 1,
                                  padIncomplete: bool = True,
                                  emitParticles: bool = False,
                                  raysPerPhoton: float | None = None,
                                  RNGSeed: int | None = None,
                                  withEventIds: bool = False,
                                  cherenkovCharges: dict[int, float] | None = None,
                                  ) -> tuple[list, int | None]:
    """
    `createTasksFromEvents` with Cherenkov light added.

    Every deposition step of a charged particle above the medium's threshold
    gains one further segment, next to its scintillation ones, whose photon
    number follows the Frank-Tamm formula.

    The emission itself comes from the media: run `buildCherenkovTables` on them
    *before* they go into `matStore`, or their Cherenkov light is dropped (with a
    warning).

    Parameters
    ----------
    events: Iterable[EnergyDepositionEvents]
        Collection of event containers to generate the tasks for.
    matStore: MaterialStore
        Material Store that is used for the simulation.
    eventsPerBatch: int, default=1
        Number of track segments per task. Note that this counts *segments*, i.e.
        one entry per scintillation component of a step plus one for its
        Cherenkov light, so a medium with two components needs three times the
        value for the same number of light paths per dispatch. The
        `ScintillationCherenkovDepositionEventSource` and the
        `BatchedSpectrumWavelengthSource` must be created with the same number.
    padIncomplete: bool, default=True
        If `True`, the last batch is padded with zero-path segments when it holds
        fewer than `eventsPerBatch`. If `False` it is discarded. Unlike the old
        grouped batching this affects only *one* batch per run.
    emitParticles: bool, default=False
        Whether to emit particles instead of rays. If `True` this will draw a
        random number of photons per segment: Gaussian around the expected value
        for scintillation, using the medium's resolution scale, Poisson for
        Cherenkov.
    raysPerPhoton: float | None, default=None
        Rays to sample per physical photon, required if `emitParticles` is
        `False`.
    RNGSeed: int | None, default=None
        The seed for the photon number RNG. Segment k always gets draw k of the
        run, so the photon numbers depend on neither the chunking of `events` nor
        `eventsPerBatch`. Cherenkov draws from its own substream, so switching it
        on leaves the scintillation photon numbers of an existing run unchanged.
    withEventIds: bool, default=False
        If `True`, every entry of `tasks` is ``(task, (groupId, eventId,
        cumulativePaths))`` instead of a bare task. `groupId` and `eventId` name
        the event each segment belongs to and `cumulativePaths` its path id
        boundary, which is what a response needs to attribute a hit back to an
        event.
    cherenkovCharges: dict[int, float] | None, default=None
        Electric charge in units of e, by absolute PDG code; updates
        `DEFAULT_CHERENKOV_CHARGES` rather than replacing it. Nuclei need no
        entry unless it is to be overridden - their Z follows from the code. An
        unlisted particle counts as neutral and radiates nothing.

    Returns
    -------
    tasks: list[dict] | list[tuple[dict, tuple]]
        List of tasks to give to the scheduler, paired with their event ids if
        `withEventIds` is `True`.
    maxBatchSize: int | None
        Largest number of light paths any of the tasks needs, or `None` if the
        tasks do not fix it. Use it as the tracer's capacity.

    See Also
    --------
    theia.scintillation.buildCherenkovTables
    theia.scintillation.createTasksFromEvents
    theia.scintillation.iterCombinedTasksFromEvents
    """
    tasks: list = []
    maxBatchSize = 0
    for item in _iter_scintillation_tasks(
        events,
        matStore,
        eventsPerBatch=eventsPerBatch,
        padIncomplete=padIncomplete,
        emitParticles=emitParticles,
        RNGSeed=RNGSeed,
        raysPerPhoton=raysPerPhoton,
        withEventIds=withEventIds,
        cherenkov=True,
        cherenkovCharges=cherenkovCharges,
    ):
        tasks.append(item)
        task = item[0] if withEventIds else item
        maxBatchSize = max(maxBatchSize, task["_batchSize"])

    return tasks, (None if maxBatchSize == 0 else maxBatchSize)


def iterCombinedTasksFromEvents(
    events: Iterable[EnergyDepositionEvents],
    matStore: MaterialStore,
    *,
    eventsPerBatch: int = 1,
    padIncomplete: bool = True,
    emitParticles: bool = False,
    raysPerPhoton: float | None = None,
    RNGSeed: int | None = None,
    withEventIds: bool = False,
    cherenkovCharges: dict[int, float] | None = None,
) -> Iterator:
    """
    `iterTasksFromEvents` with Cherenkov light added.

    See `createCombinedTasksFromEvents` for what the extra segments are, and
    `iterTasksFromEvents` for the remaining parameters.
    """
    yield from _iter_scintillation_tasks(
        events,
        matStore,
        eventsPerBatch=eventsPerBatch,
        padIncomplete=padIncomplete,
        emitParticles=emitParticles,
        RNGSeed=RNGSeed,
        raysPerPhoton=raysPerPhoton,
        withEventIds=withEventIds,
        cherenkov=True,
        cherenkovCharges=cherenkovCharges,
    )


def _make_float_array(n):
    """Creates a helper class to use GLSL float arrays in LightParams."""
    return _make_scalar_array(c_float, np.float32, n)


def _make_uint_array(n):
    """Creates a helper class to use GLSL uint arrays in LightParams."""
    return _make_scalar_array(c_uint32, np.uint32, n)


def _make_vec3_array(n):
    """
    Creates a helper class to use GLSL-Arrays in LightParams.
    """
    class Vec3Array(mat):
        _fields_ = [("items", vec3 * n)]
        _type_ = vec3
        _length_ = n
        # The `value` setter of the base class assigns element by element, which
        # is too slow for typical scintillation simulations. ctypes packs vec3 
        # into 12 bytes, so an (n, 3) float32 array has exactly this layout.
        @property
        def value(self):
            return tuple(self[i] for i in range(n))
        @value.setter
        def value(self, value):
            arr = np.ascontiguousarray(value, dtype=np.float32)
            if arr.size != 3 * n:
                raise ValueError(f"expected {n} positions, got {arr.size / 3:g}")
            memmove(addressof(self), arr.ctypes.data, sizeof(self))
    return Vec3Array


class ScintillationDepositionEventSource(LightSource):
    """
    Samples scintillation light from one or multiple energy deposition event(s). 
    It is assumed that the energy deposition on each track segment is uniform.

    Parameters
    ----------
    wavelengthSource: WavelengthSource
        Source to sample wavelengths from. Required for forward mode.
    mediumIdx: int | Iterable[int], default=0
        Index of the medium each segment is located in.
    eventsPerBatch: int, default=1
        Number of track segments stored per batch. Cannot be changed after
        initialization.
    startPosition: (float,float,float) | Iterable[(float,float,float)], default=(0,0,0)
        Start position(s) of the track segment(s).
    endPosition: (float,float,float) | Iterable[(float,float,float)], default=(0,0,0)
        End position(s) of the track segment(s).
    startTime: float | Iterable[float], default=0.0
        Time(s) at the start of the track segment(s).
    startVelocity: float | Iterable[float], default=0.0
        Particle velocity/velocities at the start of the track segment(s).
    endVelocity: float | Iterable[float], default=0.0
        Particle velocity/velocities at the end of the track segment(s).
    expectedPhotonNumber: float | Iterable[float], default=10.0
        Mean number of photons a segment emits, that is the deposited energy times 
        the yield.
    timeConstant: float | Iterable[float], default=0.0
        Time constant of the exponential decay of each segment.
    usePhotonCount: bool, default=True
        If `True` sampled radiance has units of number photons, otherwise eV.
    emitParticles: bool, default=False
        Whether to emit particles instead of rays. If `True` the number of photons
        specified in `numSamples` will be sampled for each track segment.
    numSamples: int | Iterable[int] | None, default=None
        Number of light paths to hand out per segment - photons if
        `emitParticles`, rays otherwise. The tracer's batch size must be set to the
        sum of this. This is required for particles and optional for rays. 
        
        For rays, if this is not set, each thread will importance sample a segment, 
        which will give the correct mean intensity but will make it impossible to 
        reconstruct the origin of a recorded detector hit. Setting numSamples gives
        a clear mapping between thread index and original event. Here, the contributions
        will be weighted according to `expectedPhotonNumber` such that the
        correct result is computed regardless of the numbers set in `numSamples`. For
        particles, `numSamples` will fully determine the number of photons and
        `expectedPhotonNumber` will be ignored.
    """

    def __init__(
        self,
        wavelengthSource: WavelengthSource | None = None,
        *,
        mediumIdx: int | Iterable[int] = 0,
        eventsPerBatch: int = 1,
        startPosition: tuple[float, float, float] | Iterable[tuple[float, float, float]] = (0.0, 0.0, 0.0),
        endPosition: tuple[float, float, float] | Iterable[tuple[float, float, float]] = (0.0, 0.0, 0.0),
        startTime: float | Iterable[float] = 0.0,
        startVelocity: float | Iterable[float] = 0.0,
        endVelocity: float | Iterable[float] = 0.0,
        expectedPhotonNumber: float | Iterable[float] = 10.0,
        timeConstant: float | Iterable[float] = 0.0,
        usePhotonCount: bool = True,
        emitParticles: bool = False,
        numSamples: int | Iterable[int] | None = None,
    ) -> None:
        FloatArray = _make_float_array(eventsPerBatch)
        UintArray = _make_uint_array(eventsPerBatch)
        Vec3Array = _make_vec3_array(eventsPerBatch)

        # the path index determines the segment whenever the number of paths per
        # segment is known; only then can a hit be attributed back to an event.
        # Particles always are - `numSamples` is required there - rays only if
        # they were given a count instead of being importance sampled.
        deterministic = emitParticles or numSamples is not None
        deterministicRays = not emitParticles and numSamples is not None
        fields = [
            ("startPosition",          Vec3Array),
            ("endPosition",            Vec3Array),
            ("mediumIdx",              UintArray),
            ("startTime",              FloatArray),
            ("startVelocity",          FloatArray),
            ("endVelocity",            FloatArray),
            ("timeConstant",           FloatArray),
        ]
        if deterministic:
            fields.append(("cumulativeCount", UintArray))
        else:
            fields.append(("cumulativePhotonNumber", FloatArray))
        if deterministicRays:
            fields.append(("photonsPerRay", FloatArray))
        # subclasses append their own columns here; the base adds none, so its
        # layout and its shader are exactly what they were
        fields += self._extraParamFields(FloatArray, UintArray, Vec3Array)

        class LightParams(Structure):
            _fields_ = fields

        # the extra draw is the one importance sampling the segment; a
        # deterministic assignment needs none
        addRNG = 1 if eventsPerBatch > 1 and not deterministic else 0
        extra = {"expectedPhotonNumber"}
        if numSamples is not None:
            extra.add("numSamples")
        super().__init__(
            nRNGForward=(0 if wavelengthSource is None
                         else wavelengthSource.nRNGSamples + 4 + addRNG),
            nRNGBackward=2 + addRNG,
            wavelengthSource=wavelengthSource,
            params={"LightParams": LightParams},
            extra=extra,
        )
        # check params
        if emitParticles and not usePhotonCount:
            raise ValueError("Must use photon count when emitting particles")
        if emitParticles and numSamples is None:
            raise ValueError("Particles need 'numSamples'!")

        # save params
        self._eventsPerBatch = eventsPerBatch
        self._usePhotonCount = usePhotonCount
        self._emitParticles = emitParticles
        self._deterministicRays = deterministicRays

        def _norm_vec3(value, name):
            """Normalize to a list of eventsPerBatch positions.

            A single (x, y, z) is broadcast to all batch elements; an iterable
            of positions must already have length eventsPerBatch.
            """
            if isinstance(value[0], (int, float)):
                return [tuple(value)] * eventsPerBatch
            values = list(value)
            if len(values) != eventsPerBatch:
                raise ValueError(f"'{name}' must have length {eventsPerBatch}, "
                                 f"got {len(values)}")
            return values

        def _norm_scalar(value, name):
            """Normalize to a list of eventsPerBatch values.

            A single scalar is broadcast to all batch elements; an iterable of
            values must already have length eventsPerBatch. Used for the integer
            columns too - the ctypes array setter does the narrowing.
            """
            if isinstance(value, (int, float)):
                return [value] * eventsPerBatch
            values = list(value)
            if len(values) != eventsPerBatch:
                raise ValueError(f"'{name}' must have length {eventsPerBatch}, "
                                 f"got {len(values)}")
            return values

        startPosition    = _norm_vec3(startPosition,    "startPosition")
        endPosition      = _norm_vec3(endPosition,      "endPosition")
        startTime        = _norm_scalar(startTime,       "startTime")
        startVelocity    = _norm_scalar(startVelocity,   "startVelocity")
        endVelocity      = _norm_scalar(endVelocity,     "endVelocity")
        expectedPhotonNumber = _norm_scalar(expectedPhotonNumber,
                                            "expectedPhotonNumber")
        mediumIdx        = _norm_scalar(mediumIdx,       "mediumIdx")
        timeConstant     = _norm_scalar(timeConstant,    "timeConstant")

        self.setParams(
            startPosition=startPosition,
            endPosition=endPosition,
            mediumIdx=mediumIdx,
            startTime=startTime,
            startVelocity=startVelocity,
            endVelocity=endVelocity,
            expectedPhotonNumber=expectedPhotonNumber,
            timeConstant=timeConstant,
        )
        if numSamples is not None:
            self.setParam("numSamples", _norm_scalar(numSamples, "numSamples"))

    @property
    def eventsPerBatch(self) -> int:
        """The number of track segments in each batch."""
        return self._eventsPerBatch
    
    @property
    def emitsParticles(self) -> bool:
        """Whether this light source emits particles instead of light rays"""
        return self._emitParticles

    @property
    def deterministicRays(self) -> bool:
        """Whether rays are handed out per segment via 'numSamples'"""
        return self._deterministicRays

    @property
    def deterministicSegments(self) -> bool:
        """
        Whether the segment a path belongs to follows from its index alone.
        """
        return self._emitParticles or self._deterministicRays

    @property
    def usePhotonCount(self) -> bool:
        """Whether to sample radiance in eV or #photons"""
        return self._usePhotonCount

    @property
    def forwardSourceCode(self) -> str | None:
        if self.wavelengthSource is None:
            return None
        preamble = createPreamble(
            EVENTS_PER_BATCH=self.eventsPerBatch,
            LIGHT_SOURCE_EMIT_PARTICLE=self.emitsParticles,
            LIGHT_SOURCE_DETERMINISTIC_RAYS=self.deterministicRays,
            SCINTILLATION_USE_ENERGY=not self.usePhotonCount,
            **self._extraDefines(),
        )
        code = loadShader("lightsource/scintillation/deposition_event/forward.glsl")
        return preamble + code

    @property
    def backwardSourceCode(self) -> str | None:
        if self.emitsParticles:
            return None
        preamble = createPreamble(
            EVENTS_PER_BATCH=self._eventsPerBatch,
            LIGHT_SOURCE_EMIT_PARTICLE=self.emitsParticles,
            LIGHT_SOURCE_DETERMINISTIC_RAYS=self.deterministicRays,
            SCINTILLATION_USE_ENERGY=not self.usePhotonCount,
            **self._extraDefines(),
        )
        code = loadShader("lightsource/scintillation/deposition_event/backward.glsl")
        return preamble + code
    
    def _extraParamFields(self, FloatArray, UintArray, Vec3Array) -> list:
        """Extra columns a subclass wants in `LightParams`, appended last."""
        return []

    def _extraDefines(self) -> dict:
        """Extra preamble defines a subclass needs for those columns."""
        return {}

    def _finishParams(self, i: int) -> None:
        super()._finishParams(i)
        if self.deterministicSegments:
            pathCount = np.asarray(self.getParam("numSamples"), np.uint64)
            cumulativePaths = np.cumsum(pathCount)
            if len(cumulativePaths) and cumulativePaths[-1] > np.iinfo(np.uint32).max:
                raise ValueError(
                    f"{cumulativePaths[-1]} paths in one batch exceed the "
                    f"{np.iinfo(np.uint32).max} the shader can index")
            self.setParam("cumulativeCount", cumulativePaths)
            if self.deterministicRays:
                # each ray stands for the photons of its own segment
                expectedPhotons = np.asarray(
                    self.getParam("expectedPhotonNumber"), np.float64)
                self.setParam("photonsPerRay", np.divide(
                    expectedPhotons, pathCount,
                    out=np.zeros_like(expectedPhotons), where=pathCount > 0))
        else:
            # compute culmulative expected number of photons for sampling
            self.setParam("cumulativePhotonNumber", np.cumsum(
                np.asarray(self.getParam("expectedPhotonNumber"), np.float64)))


class ScintillationCherenkovDepositionEventSource(ScintillationDepositionEventSource):
    """
    `ScintillationDepositionEventSource` that also emits Cherenkov light.

    Segments carry one extra column, `mode`, telling the shader which kind of
    light a segment stands for: scintillation emits isotropically and delayed by
    the component's time constant, Cherenkov emits promptly onto a cone about the
    segment direction with ``cos(theta) = 1 / (beta * n(lambda))``.

    Parameters
    ----------
    mode: int | Iterable[int], default=0
        Per segment, 0 for scintillation and 1 for Cherenkov. 
    *args, **kwargs
        As `ScintillationDepositionEventSource`.

    Stage Parameters
    ----------------
    mode: int | Iterable[int]
        Per segment, 0 for scintillation and 1 for Cherenkov.

    See Also
    --------
    theia.scintillation.ScintillationDepositionEventSource
    theia.scintillation.createCombinedTasksFromEvents
    """

    name = "Scintillation and Cherenkov Deposition Event Light Source"

    def __init__(self, *args, mode: int | Iterable[int] = 0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        n = self.eventsPerBatch
        values = [mode] * n if isinstance(mode, (int, float)) else list(mode)
        if len(values) != n:
            raise ValueError(f"'mode' must have length {n}, got {len(values)}")
        self.setParam("mode", values)

    def _extraParamFields(self, FloatArray, UintArray, Vec3Array) -> list:
        return [("mode", UintArray)]

    def _extraDefines(self) -> dict:
        return {"LIGHT_SOURCE_CHERENKOV": True}

    @property
    def backwardSourceCode(self) -> str | None:
        """
        `None`: backward tracing is not supported here.
        """
        return None
