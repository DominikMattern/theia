from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from collections.abc import Iterable, Iterator
from warnings import warn
from ctypes import Structure, c_float, c_uint32
from hephaistos.glsl import vec3, mat, vec
from theia.compiler import createPreamble, loadShader

from theia.light import LightSource, WavelengthSource
from theia.material import Medium, MaterialStore
import theia.units as u

__all__ = [
    "EnergyDepositionEvents",
    "ScintillationDepositionEventSource",
    "createTasksFromEvents",
    "iterTasksFromEvents",
]


def __dir__():
    return __all__


@dataclass
class EnergyDepositionEvents:
    """
    Data class storing energy deposition events in a form that is compatible with 
    the output of remage simulations.

    Each instance describes all deposition steps within a single `medium`. Every
    per-step quantity is stored as a NumPy array of equal length, with the k-th
    entry across all arrays describing the k-th deposition step. The start and
    end positions are split into separate arrays for each coordinate.
    """
    eventId: np.ndarray
    medium: Medium
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


_ZERO_STEP = {
    'startPosition': (0.0, 0.0, 0.0),
    'endPosition':   (0.0, 0.0, 0.0),
    'startTime':      0.0,
    'startVelocity':  0.0,
    'endVelocity':    0.0,
    'energyDeposition': 0.0,
}


def _fetch_scintillation_property(medium: Medium, generic_key: str, code: int):
    """
    Return the medium property for `generic_key`, preferring the
    particle-specific variant ``f"{generic_key}_{code}"`` if present.

    Nuclear PDG codes (ions, i.e. ``code > 1000000000`` such as alphas,
    deuterons and heavy nuclei) fall back to a shared ``f"{generic_key}_ion"``
    variant before the generic property. Returns the generic property, or 
    `None`, if none of the above exists.
    """
    specific = f"{generic_key}_{code}"
    if specific in medium.properties:
        return medium.properties[specific]
    if code > 1000000000:
        ion_key = f"{generic_key}_ion"
        if ion_key in medium.properties:
            return medium.properties[ion_key]
    return medium.properties.get(generic_key)


def _fetch_component_props(medium: Medium, particleCode: int, component: int):
    """
    Fetch the scintillation properties of `medium` for one particle type and
    scintillation component.

    Particle-dependent properties are keyed by the *absolute* PDG code appended
    as a suffix (e.g. ``scintillation_yield_11`` applies to electrons and
    positrons); if no particle-specific variant is defined, the generic property
    (without suffix) is used as fallback. Returns ``(branchingRatio,
    photonYield, timeConstant, spectrumSampler, resolutionScale)`` or `None` if
    the medium defines no scintillation for this component (i.e. the particle
    does not scintillate here).
    """
    code = abs(int(particleCode))
    bR = _fetch_scintillation_property(medium, f"scintillation_branching_ratio_{component}", code)
    pY = _fetch_scintillation_property(medium, "scintillation_yield", code)
    tC = _fetch_scintillation_property(medium, f"scintillation_time_constant_{component}", code)
    spec = _fetch_scintillation_property(medium, f"scintillation_spectrum_{component}_sampler", code)
    if bR is None or pY is None or tC is None or spec is None:
        return None
    rs = _fetch_scintillation_property(medium, "scintillation_resolutionscale", code)
    resolutionScale = rs.value[1] if rs is not None else 1.0
    return bR.value[1], pY.value[1], tC.value[1], spec.data, resolutionScale


def _resolve_group_adds(medium: Medium, medIdx: int, particleCode: int):
    """
    Resolve the active scintillation components for a single deposition step.

    Returns a list of ``(group_key, groupInfo)`` entries to accumulate, where
    ``groupInfo = (medIdx, effectiveYield, timeConstant, spectrum,
    resolutionScale)``. Empty if the particle does not scintillate in `medium`.
    """
    props = _fetch_component_props(medium, particleCode, 1)
    if props is None:
        return []
    bR1, pY1, tC1, spec1, rs = props
    adds = [((medium.name, 1, particleCode), (medIdx, pY1 * bR1, tC1, spec1, rs))]

    if 1.0 - bR1 > 1e-5:
        props = _fetch_component_props(medium, particleCode, 2)
        if props is None:
            warn(f"Medium '{medium.name}': branching ratio < 1 for particle "
                 f"{particleCode} but no scintillation component 2 defined; "
                 "ignoring the remaining light.")
            return adds
        bR2, pY2, tC2, spec2, _ = props
        adds.append(((medium.name, 2, particleCode), (medIdx, pY2 * bR2, tC2, spec2, rs)))

        if 1.0 - bR1 - bR2 > 1e-5:
            props = _fetch_component_props(medium, particleCode, 3)
            if props is None:
                warn(f"Medium '{medium.name}': branching ratio < 1 for particle "
                     f"{particleCode} but no scintillation component 3 defined; "
                     "ignoring the remaining light.")
                return adds
            bR3, pY3, tC3, spec3, _ = props
            adds.append(((medium.name, 3, particleCode), (medIdx, pY3 * bR3, tC3, spec3, rs)))

    return adds


def _iter_scintillation_tasks(events, matStore, *, eventsPerBatch, padIncomplete,
                              emitParticles, seed_sequence):
    """
    Core generator shared by `createTasksFromEvents` and `iterTasksFromEvents`.

    Groups deposition steps by (medium, component, particle code), keeping the
    accumulators alive across the whole `events` iterable, and yields one task
    dict per completed batch. Incomplete trailing batches are padded (or
    dropped, if `padIncomplete` is `False`) only once the iterable is exhausted,
    so batching is independent of how `events` is chunked.
    """
    accumulators: dict = {}   # group_key -> list of pending per-step dicts
    group_info: dict = {}     # group_key -> (medIdx, effYield, timeConstant, spectrum, resolutionScale)

    def _make_task(group_key, batch, pad_to_full=False):
        medIdx, eff_yield, tc, spec, resolutionscale = group_info[group_key]
        if pad_to_full:
            batch = batch + [_ZERO_STEP] * (eventsPerBatch - len(batch))
        task = {
            "lightSource__mediumIdx":        medIdx,
            "lightSource__startPosition":    [e['startPosition']    for e in batch],
            "lightSource__endPosition":      [e['endPosition']      for e in batch],
            "lightSource__startTime":        [e['startTime']        for e in batch],
            "lightSource__startVelocity":    [e['startVelocity']    for e in batch],
            "lightSource__endVelocity":      [e['endVelocity']      for e in batch],
            "lightSource__timeConstant":     tc,
            "photons__spectrumTableAddress": spec,
        }
        if emitParticles:
            rng = np.random.default_rng(seed_sequence.spawn(1)[0])
            rdm = rng.normal(0, 1, eventsPerBatch)
            expected_number = np.asarray([e['energyDeposition'] for e in batch]) * eff_yield
            number = np.clip(
                np.round(expected_number + np.sqrt(expected_number) * resolutionscale * rdm),
                0, None,
            )
            task["lightSource__numPhotons"] = list(number)
            task["_batchSize"] = int(np.sum(number))
        else:
            task["lightSource__energyDeposition"] = [e['energyDeposition'] for e in batch]
            task["lightSource__photonYield"] = eff_yield
        return task

    for evts in events:
        medium = evts.medium
        medIdx = matStore.media[medium.name]

        for k in range(len(evts)):
            adds = _resolve_group_adds(medium, medIdx, int(evts.particleCode[k]))
            if not adds:
                continue

            edict = {
                'startPosition':    (evts.startPositionX[k], evts.startPositionY[k], evts.startPositionZ[k]),
                'endPosition':      (evts.endPositionX[k], evts.endPositionY[k], evts.endPositionZ[k]),
                'startTime':        evts.startTime[k],
                'startVelocity':    evts.startVelocity[k],
                'endVelocity':      evts.endVelocity[k],
                'energyDeposition': evts.energyDeposition[k],
            }

            for group_key, info in adds:
                if group_key not in accumulators:
                    accumulators[group_key] = []
                    group_info[group_key] = info
                accumulators[group_key].append(edict)
                if len(accumulators[group_key]) == eventsPerBatch:
                    yield _make_task(group_key, accumulators.pop(group_key))

    for group_key, batch in accumulators.items():
        if padIncomplete:
            yield _make_task(group_key, batch, pad_to_full=True)
        # else: discard incomplete last batch


def createTasksFromEvents(events: Iterable[EnergyDepositionEvents],
                          matStore: MaterialStore,
                          *,
                          eventsPerBatch: int = 1,
                          padIncomplete: bool = True,
                          emitParticles: bool = False,
                          RNGSeed: int | None = None,
                          ) -> tuple[list[dict], int | None]:
    """
    Utility function to prepare the simulation of scintillation light. Takes
    a list of events and returns a list of tasks which properly configures
    the light source and wavelength source for a scheduler, assuming a
    `ScintillationDepositionEventSource` and a `SpectrumWavelengthSource` is used.
    Scintillation spectrums, time constants, branching ratios and yields must be
    defined for all scintillating media through their medium properties (see
    `material.OpticalProperties` for the property names).

    Parameters
    ----------
    events: Iterable[EnergyDepositionEvents]
        Collection of event containers to generate the tasks for. Each container 
        holds all deposition steps for a single medium.
    matStore: MaterialStore
        Material Store that is used for the simulation.
    eventsPerBatch: int, default=1
        Number of track segments per task. Events are grouped into batches by
        (medium, scintillation component, particle code) — all three must match
        within a batch. The `ScintillationDepositionEventSource` must be
        created with the same eventsPerBatch.
    padIncomplete: bool, default=True
        If `True`, the last batch of each group is padded with zero-energy
        segments when it has fewer than batchSize events. If `False`,
        incomplete last batches are discarded.
    emitParticles: bool, default=False
        Whether to emit particles instead of rays. If `True` this will draw a
        random number of photons per event based on the expected value and the
        madium's resolution scale.
    RNGSeed: int | None, default=None
        The seed for the photon number RNG.
    
    Returns
    -------
    tasks: list[dict]
        List of tasks to give to the scheduler.
    maxBatchSize: int | None
        If emitParticles is `True`, this gives the size of the largest batch.

    """
    seed_sequence = np.random.SeedSequence(RNGSeed)
    tasks: list[dict] = []
    maxBatchSize = 0
    for task in _iter_scintillation_tasks(
        events,
        matStore,
        eventsPerBatch=eventsPerBatch,
        padIncomplete=padIncomplete,
        emitParticles=emitParticles,
        seed_sequence=seed_sequence,
    ):
        tasks.append(task)
        if emitParticles:
            maxBatchSize = max(maxBatchSize, task["_batchSize"])

    return tasks, (None if maxBatchSize == 0 else maxBatchSize)


def iterTasksFromEvents(events: Iterable[EnergyDepositionEvents],
                        matStore: MaterialStore,
                        *,
                        eventsPerBatch: int = 1,
                        padIncomplete: bool = True,
                        emitParticles: bool = False,
                        RNGSeed: int | None = None,
                        ) -> Iterator[dict]:
    """
    Streaming variant of `createTasksFromEvents` yielding tasks lazily.

    Behaves exactly like `createTasksFromEvents` (identical grouping, batching
    and task layout) but returns a generator instead of a list. The per-group
    accumulators are kept alive across the whole `events` iterable, so `events`
    may itself be a generator producing chunks (e.g. read from disk on demand)
    without changing the batching compared to processing everything at once.

    This enables overlapping event reading / task building with GPU work and
    keeps host memory bounded. The trade-off is that `maxBatchSize` is not known
    up front: when `emitParticles` is `True` the tracer capacity must therefore
    be fixed a priori (each yielded task still carries its own `_batchSize`).

    Parameters
    ----------
    events: Iterable[EnergyDepositionEvents]
        Collection of event containers to generate the tasks for. Each container 
        holds all deposition steps for a single medium.
    matStore: MaterialStore
        Material Store that is used for the simulation.
    eventsPerBatch: int, default=1
        Number of track segments per task. Events are grouped into batches by
        (medium, scintillation component, particle code) — all three must match
        within a batch. The `ScintillationDepositionEventSource` must be
        created with the same eventsPerBatch.
    padIncomplete: bool, default=True
        If `True`, the last batch of each group is padded with zero-energy
        segments when it has fewer than batchSize events. If `False`,
        incomplete last batches are discarded.
    emitParticles: bool, default=False
        Whether to emit particles instead of rays. If `True` this will draw a
        random number of photons per event based on the expected value and the
        madium's resolution scale.
    RNGSeed: int | None, default=None
        The seed for the photon number RNG.

    Yields
    ------
    task: dict
        One task per completed batch, ready to hand to a scheduler.

    See Also
    --------
    theia.scintillation.createTasksFromEvents
    """
    seed_sequence = np.random.SeedSequence(RNGSeed)
    yield from _iter_scintillation_tasks(
        events,
        matStore,
        eventsPerBatch=eventsPerBatch,
        padIncomplete=padIncomplete,
        emitParticles=emitParticles,
        seed_sequence=seed_sequence,
    )


def _make_float_array(n):
    """
    Creates a helper class to use GLSL-Arrays in LightParams.
    """
    def _initVecFields(scalar, dim):
        return [(f"Component_{d}", scalar) for d in range(dim)]
    class FloatArray(vec):
        _fields_ = _initVecFields(c_float, n)
        _scalar_ = c_float
        _dim_ = n
        # overwrite functions to access the Component_<idx> fields directly
        # instead of the x/y/z/w swizzling of the base vec class
        def __getitem__(self, idx):
            if isinstance(idx, slice):
                return tuple(self[i] for i in range(self._dim_)[idx])
            elif idx >= self._dim_:
                raise IndexError("vector has no such dimension")
            else:
                return getattr(self, f"Component_{idx}")
        def __getattr__(self, name: str):
            # only called when normal lookup fails; the ctypes fields resolve
            # via their descriptors, so anything reaching here is truly missing
            raise AttributeError(name)
        def __setitem__(self, idx, value):
            if isinstance(idx, slice):
                for i in range(self._dim_)[idx]:
                    try:
                        self[i] = value[i]
                    except:
                        self[i] = value
            elif idx >= self._dim_:
                raise IndexError("vector has no such dimension")
            else:
                setattr(self, f"Component_{idx}", value)
        def __setattr__(self, name: str, value):
            # bypass vec.__setattr__ (which swizzles multi-char names) and let
            # ctypes store the field directly
            Structure.__setattr__(self, name, value)
    return FloatArray


def _make_vec3_array(n):
    """
    Creates a helper class to use GLSL-Arrays in LightParams.
    """
    class Vec3Array(mat):
        _fields_ = [("items", vec3 * n)]
        _type_ = vec3
        _length_ = n
    return Vec3Array


class ScintillationDepositionEventSource(LightSource):
    """
    Samples scintillation light from one or multiple energy deposition event(s). 
    It is assumed that the energy deposition on each track segment is uniform.

    Parameters
    ----------
    wavelengthSource: WavelengthSource
        Source to sample wavelengths from. Required for forward mode.
    mediumIdx: int, default=0
        Index of the medium the light source is located in.
    eventsPerBatch: int, default=1
        Number of track segments stored per batch. Each simulation step
        picks one segment at random. Cannot be changed after initialization.
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
    energyDeposition: float | Iterable[float], default=1.0 keV
        Energy deposited per segment. Will be ignored if `emitParticles` is `True`.
    photonYield: float, default=10000/MeV
        Number of generated photons per deposited energy. Will be ignored if 
        `emitParticles` is `True`.
    timeConstant: float, default=0.0
        Time constant of the exponential decay.
    usePhotonCount: bool, default=True
        If `True` sampled radiance has units of number photons, otherwise eV.
    emitParticles: bool, default=False
        Whether to emit particles instead of rays. If `True` the number of photons
        specified in `numPhotons` will be sampled for each track segment.
    numPhotons: int | Iterable[int], default=1
        Number of photons to sample per event. The sum of these in a batch must
        be equal to the batch size of the tracer. Will be ignored if `emitParticles`
        is `False`.
    """

    def __init__(
        self,
        wavelengthSource: WavelengthSource | None = None,
        *,
        mediumIdx: int = 0,
        eventsPerBatch: int = 1,
        startPosition: tuple[float, float, float] | Iterable[tuple[float, float, float]] = (0.0, 0.0, 0.0),
        endPosition: tuple[float, float, float] | Iterable[tuple[float, float, float]] = (0.0, 0.0, 0.0),
        startTime: float | Iterable[float] = 0.0,
        startVelocity: float | Iterable[float] = 0.0,
        endVelocity: float | Iterable[float] = 0.0,
        energyDeposition: float | Iterable[float] = 1.0 * u.keV,
        photonYield: float = 10 / u.keV,
        timeConstant: float = 0.0,
        usePhotonCount: bool = True,
        emitParticles: bool = False,
        numPhotons: int | Iterable[int] = 1,
    ) -> None:
        _FA = _make_float_array(eventsPerBatch)
        _V3A = _make_vec3_array(eventsPerBatch)

        class LightParams(Structure):
            _fields_ = [
                ("startPosition",          _V3A),
                ("endPosition",            _V3A),
                ("mediumIdx",              c_uint32),
                ("startTime",              _FA),
                ("startVelocity",          _FA),
                ("endVelocity",            _FA),
                ("cumulativePhotonNumber", _FA),
                ("timeConstant",           c_float),
            ]

        lam = wavelengthSource
        addRNG = 1 if eventsPerBatch > 1 and not emitParticles else 0
        super().__init__(
            nRNGForward=0 if lam is None else lam.nRNGSamples + 4 + addRNG,
            nRNGBackward=2 + addRNG,
            wavelengthSource=wavelengthSource,
            params={"LightParams": LightParams},
            extra={"energyDeposition", "photonYield", "numPhotons"}
        )
        # check params
        if emitParticles and not usePhotonCount:
            raise ValueError("Must use photon count when emitting particles")

        # save params
        self._batchSize = eventsPerBatch
        self._usePhotonCount = usePhotonCount
        self._emitParticles = emitParticles

        def _norm_vec3(v, name):
            """Normalize to a list of batchSize positions.

            A single (x, y, z) is broadcast to all batch elements; an iterable
            of positions must already have length batchSize.
            """
            if isinstance(v[0], (int, float)):
                return [tuple(v)] * eventsPerBatch
            v = list(v)
            if len(v) != eventsPerBatch:
                raise ValueError(f"'{name}' must have length {eventsPerBatch}, got {len(v)}")
            return v

        def _norm_float(v, name):
            """Normalize to a list of batchSize values.

            A single scalar is broadcast to all batch elements; an iterable of
            values must already have length batchSize.
            """
            if isinstance(v, (int, float)):
                return [v] * eventsPerBatch
            v = list(v)
            if len(v) != eventsPerBatch:
                raise ValueError(f"'{name}' must have length {eventsPerBatch}, got {len(v)}")
            return v

        startPosition    = _norm_vec3(startPosition,    "startPosition")
        endPosition      = _norm_vec3(endPosition,      "endPosition")
        startTime        = _norm_float(startTime,       "startTime")
        startVelocity    = _norm_float(startVelocity,   "startVelocity")
        endVelocity      = _norm_float(endVelocity,     "endVelocity")
        energyDeposition = _norm_float(energyDeposition,"energyDeposition")
        numPhotons       = _norm_float(numPhotons,      "numPhotons")

        self.setParams(
            startPosition=startPosition,
            endPosition=endPosition,
            mediumIdx=mediumIdx,
            startTime=startTime,
            startVelocity=startVelocity,
            endVelocity=endVelocity,
            energyDeposition=energyDeposition,
            photonYield=photonYield,
            timeConstant=timeConstant,
            numPhotons=numPhotons,
        )

    @property
    def batchSize(self) -> int:
        """The number of track segments in each batch."""
        return self._batchSize
    
    @property
    def emitsParticles(self) -> bool:
        """Whether this light source emits particles instead of light rays"""
        return self._emitParticles

    @property
    def usePhotonCount(self) -> bool:
        """Whether to sample radiance in eV or #photons"""
        return self._usePhotonCount

    @property
    def forwardSourceCode(self) -> str | None:
        if self.wavelengthSource is None:
            return None
        preamble = createPreamble(
            BATCH_SIZE=self.batchSize,
            LIGHT_SOURCE_EMIT_PARTICLE=self.emitsParticles,
            SCINTILLATION_USE_ENERGY=not self.usePhotonCount,
        )
        code = loadShader("lightsource/scintillation/deposition_event/forward.glsl")
        return preamble + code

    @property
    def backwardSourceCode(self) -> str | None:
        if self.emitsParticles:
            return None
        preamble = createPreamble(
            BATCH_SIZE=self._batchSize,
            LIGHT_SOURCE_EMIT_PARTICLE=self.emitsParticles,
            SCINTILLATION_USE_ENERGY=not self.usePhotonCount,
        )
        code = loadShader("lightsource/scintillation/deposition_event/backward.glsl")
        return preamble + code
    
    def _finishParams(self, i: int) -> None:
        super()._finishParams(i)
        if self.emitsParticles:
            numPhotons = self.getParam("numPhotons")
            # compute culmulative number of photons for sampling
            c = np.cumsum(numPhotons)
            self.setParam("cumulativePhotonNumber", c)
        else:
            photonYield = self.getParam("photonYield")
            energyDeposition = self.getParam("energyDeposition")
            # compute culmulative expected number of photons for sampling
            c = np.cumsum(np.asarray(energyDeposition) * photonYield)
            self.setParam("cumulativePhotonNumber", c)
