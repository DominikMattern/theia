from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from collections.abc import Iterable, Iterator
from warnings import warn
from ctypes import Structure, addressof, c_float, c_uint32, memmove, sizeof
from hephaistos.glsl import vec3, mat
from scipy.special import ndtri
from theia.compiler import createPreamble, loadShader

from theia.light import LightSource, WavelengthSource, _make_scalar_array
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


def _scintillation_components(medium: Medium, particleCode: int):
    """
    The scintillation components one particle type excites in `medium`.

    Returns a list of ``(component, effectiveYield, timeConstant, spectrum,
    resolutionScale)`` entries, one per active component, empty if the particle
    does not scintillate here. `_iter_scintillation_tasks` turns each of them into
    a segment of its own, so the properties travel with the segment instead of
    selecting which batch it lands in.

    Components are taken in order until their branching ratios add up to one, so a
    medium defining a single component is as valid as one defining three. Running
    out of components before reaching one warns and drops the remaining light.

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
    one fallback because they have no single PDG code.

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
        components.append((
            component,
            photonYield.value[1] * bR.value[1],
            tC.value[1],
            spectrum.data,
            rs.value[1] if rs is not None else 1.0,
        ))
        branching += bR.value[1]
        if 1.0 - branching <= 1e-5:
            break

    return components


def _expand_segments(chunk, matStore, componentCache, rng, *,
                     emitParticles, raysPerPhoton):
    """
    Turns one input chunk into a stream of segments, in input order.

    A deposition step excites up to three scintillation components, each emitting
    with its own yield, time constant and spectrum. They become *consecutive*
    segments of the stream, so all segments of a step - and hence of an event -
    stay adjacent.

    Everything is whole array work. The component set of a step follows from its
    (medium, particle code) pair alone, so the properties are resolved once per
    pair rather than once per step, and the expansion itself is one `np.repeat`
    plus one gather per column.

    Returns a dict of all segment columns, or `None` if the chunk contributes no
    scintillation light at all.
    """

    # We will use the following convention for variable names:
    #   Step: One entry of EnergyDepositionEvents
    #   Segment: One entry of ScintillationDepositionEventSource's inputs
    #   Component: One of (up to 3) scintillation components of a given medium
    #   Pair: The combination (medium, particle code) that determines the scintillation
    #         properties
    # Each step will lead to [number of components] segments.

    numSteps = len(chunk.eventId)
    if numSteps == 0:
        return None

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
    # the (medium, particle code) pair of every step, as a compact index into the
    # handful of pairs a chunk actually contains
    pairKeys, pairOfStep = np.unique(mediumOfStep * len(codes) + codeOfStep,
                                     return_inverse=True)

    # scintillation properties per pair - resolved once, then memoized across
    # chunks in `componentCache`
    componentsOfPair = []
    for key in pairKeys:
        mediumSlot, codeSlot = divmod(int(key), len(codes))
        # keyed by name, not identity: an id() can be recycled once a Medium is
        # collected, and a name is how media are identified everywhere else
        cacheKey = (media[mediumSlot].name, int(codes[codeSlot]))
        if cacheKey not in componentCache:
            componentCache[cacheKey] = _scintillation_components(
                media[mediumSlot], int(codes[codeSlot]))
        componentsOfPair.append(componentCache[cacheKey])
    maxComponents = max((len(c) for c in componentsOfPair), default=0)
    if maxComponents == 0:
        return None

    # one row per pair, padded to the widest component set, so that a segment can
    # pick its properties with a single (pair, component) gather
    numPairs = len(pairKeys)
    numComponentsOfPair = np.array([len(c) for c in componentsOfPair], np.intp)
    yieldTable = np.zeros((numPairs, maxComponents))
    timeConstantTable = np.zeros((numPairs, maxComponents), np.float32)
    spectrumTable = np.zeros((numPairs, maxComponents), np.uint64)
    resolutionScaleTable = np.ones((numPairs, maxComponents))
    for pair, components in enumerate(componentsOfPair):
        for slot, (_, effYield, tConst, spectrum, resScale) in enumerate(components):
            yieldTable[pair, slot] = effYield
            timeConstantTable[pair, slot] = tConst
            spectrumTable[pair, slot] = spectrum
            resolutionScaleTable[pair, slot] = resScale
        # the padded slots are never dereferenced - the shader's search cannot
        # land on a segment owning no path - but a null table address would be a
        # device fault if it ever were, so point them at a real table
        if components:
            spectrumTable[pair, len(components):] = components[-1][3]
    mediumIdxOfPair = storeIdxOfMedium[pairKeys // len(codes)]

    # expand: one segment per (step, component)
    numComponentsOfStep = numComponentsOfPair[pairOfStep]
    numSegments = int(numComponentsOfStep.sum())
    if numSegments == 0:
        return None
    stepOfSegment = np.repeat(np.arange(numSteps), numComponentsOfStep)
    componentOfSegment = np.arange(numSegments) - np.repeat(
        np.cumsum(numComponentsOfStep) - numComponentsOfStep, numComponentsOfStep)
    pairOfSegment = pairOfStep[stepOfSegment]

    def gatherPosition(x, y, z):
        # one pass, no np.stack().astype() intermediate
        out = np.empty((numSegments, 3), np.float32)
        out[:, 0] = np.asarray(x, np.float32)[stepOfSegment]
        out[:, 1] = np.asarray(y, np.float32)[stepOfSegment]
        out[:, 2] = np.asarray(z, np.float32)[stepOfSegment]
        return out

    segments = {
        "startPosition": gatherPosition(chunk.startPositionX,
                                        chunk.startPositionY,
                                        chunk.startPositionZ),
        "endPosition": gatherPosition(chunk.endPositionX,
                                      chunk.endPositionY,
                                      chunk.endPositionZ),
        "mediumIdx": mediumIdxOfPair[pairOfSegment],
        "startTime": np.asarray(chunk.startTime, np.float32)[stepOfSegment],
        "startVelocity": np.asarray(chunk.startVelocity, np.float32)[stepOfSegment],
        "endVelocity": np.asarray(chunk.endVelocity, np.float32)[stepOfSegment],
        "timeConstant": timeConstantTable[pairOfSegment, componentOfSegment],
        "energyDeposition": np.asarray(chunk.energyDeposition,
                                       np.float64)[stepOfSegment],
        "photonYield": yieldTable[pairOfSegment, componentOfSegment],
        "spectrum": spectrumTable[pairOfSegment, componentOfSegment],
        "eventId": np.asarray(chunk.eventId, np.int64)[stepOfSegment],
        "groupId": np.asarray(chunk.groupId, np.int64)[stepOfSegment],
    }

    expectedPhotons = segments["energyDeposition"] * segments["photonYield"]
    if emitParticles:
        # One draw per segment, taken here rather than per task so that segment k
        # always gets draw k of the run: `random` consumes exactly one 64 bit word
        # per value, so the stream is position addressable and the photon numbers
        # do not depend on how `events` is chunked nor on `eventsPerBatch`.
        # `ndtri` is the exact inverse normal CDF; the floor keeps a (vanishingly
        # unlikely) exact zero from turning into -inf.
        uniform = np.maximum(rng.random(numSegments), 5e-324)
        resolutionScale = resolutionScaleTable[pairOfSegment, componentOfSegment]
        pathCount = np.rint(expectedPhotons
                            + np.sqrt(expectedPhotons) * resolutionScale
                            * ndtri(uniform))
    else:
        # a fixed number of rays per segment, so that the segment - and with it
        # the event - follows from the path index
        pathCount = np.where(
            expectedPhotons > 0.0,
            np.maximum(1.0, np.rint(expectedPhotons * raysPerPhoton)), 0.0)
    segments["pathCount"] = np.clip(pathCount, 0, None).astype(np.uint64)
    return segments


def _iter_scintillation_tasks(events, matStore, *, eventsPerBatch, padIncomplete,
                              emitParticles, RNGSeed, raysPerPhoton=None,
                              withEventIds=False):
    """
    Core generator shared by `createTasksFromEvents` and `iterTasksFromEvents`.

    Emits tasks strictly in input order: the deposition steps of a chunk are
    expanded into segments (`_expand_segments`), appended to a single stream, and
    cut into batches of `eventsPerBatch`. A segment's properties travel with it,
    so one batch may mix scintillation components, particle types and media - the
    price for having the hits of an event land next to each other in the output.

    A complete batch is a contiguous *view* into the expanded arrays; only the
    remainder crossing a chunk boundary is ever copied.
    """
    # Reject a ray mode that could not attribute its paths to a segment. Rays
    # need a fixed count per segment, since tasks in input order require the
    # segment of a path to follow from its index - which importance sampling one
    # segment per ray cannot provide.
    if emitParticles:
        if raysPerPhoton is not None:
            raise ValueError("'raysPerPhoton' is for rays; particles always get "
                             "one path per photon")
    else:
        if raysPerPhoton is None:
            raise ValueError(
                "rays need 'raysPerPhoton': tasks in input order require the "
                "segment of a path to follow from its index, which importance "
                "sampling one segment per ray cannot provide")
        if not raysPerPhoton > 0.0:
            raise ValueError(
                f"'raysPerPhoton' must be positive, got {raysPerPhoton}")
        raysPerPhoton = float(raysPerPhoton)

    # One generator for the whole run, so segment k always gets draw k. Philox
    # and `random()` on purpose: `random()` consumes exactly one 64 bit word per
    # value, so the stream is position addressable and segment k's photon number
    # can be recomputed on its own. Philox emits four words per counter block, 
    # so the jump to draw k is `advance(k // 4)` plus `k % 4` discards. 
    rng = np.random.Generator(np.random.Philox(key=RNGSeed))

    componentCache: dict = {}    # (mediumName, particleCode) -> components
    pending: dict | None = None  # < eventsPerBatch segments, owning their memory

    def make_task(segments):
        cumulativePaths = np.cumsum(segments["pathCount"])
        numPaths = int(cumulativePaths[-1]) if len(cumulativePaths) else 0
        if numPaths > np.iinfo(np.uint32).max:
            raise ValueError(
                f"a batch would hold {numPaths} paths, more than the "
                f"{np.iinfo(np.uint32).max} the shader can index; lower "
                "eventsPerBatch or raysPerPhoton")
        task = {
            f"lightSource__{name}": segments[name]
            for name in ("startPosition", "endPosition", "mediumIdx", "startTime",
                         "startVelocity", "endVelocity", "timeConstant",
                         "energyDeposition", "photonYield")
        }
        # the light source hands out this many paths per segment - photons for
        # particles, rays otherwise - and the wavelength source needs the same
        # numbers to find the same segment
        task["lightSource__numSamples"] = segments["pathCount"]
        task["photons__spectrumTableAddress"] = segments["spectrum"]
        task["photons__count"] = segments["pathCount"]
        task["_batchSize"] = numPaths
        # (params, args): the scheduler hands `args` to the process callback, so a
        # response can attribute its hits back to events. `cumulativePaths` is the
        # path id boundary of every segment, `(groupId, eventId)` says which event
        # of which group owns it.
        return ((task, (segments["groupId"], segments["eventId"], cumulativePaths))
                if withEventIds else task)

    for chunk in events:
        segments = _expand_segments(chunk, matStore, componentCache, rng,
                                    emitParticles=emitParticles,
                                    raysPerPhoton=raysPerPhoton)
        if segments is None:
            continue
        if pending is not None:
            segments = {name: np.concatenate((pending[name], segments[name]))
                        for name in ("startPosition", "endPosition", "mediumIdx", 
                                     "startTime", "startVelocity", "endVelocity", 
                                     "timeConstant", "energyDeposition", "photonYield", 
                                     "spectrum",   "pathCount", "eventId", "groupId")
            }
            pending = None
        numSegments = len(segments["startTime"])
        numComplete = numSegments - numSegments % eventsPerBatch
        for start in range(0, numComplete, eventsPerBatch):
            yield make_task({name: column[start:start + eventsPerBatch]
                             for name, column in segments.items()})
        # The remainder carries over to the next chunk and must own its memory:
        # the tasks handed out above are consumed asynchronously, so it may
        # neither alias nor pin their arrays.
        if numComplete < numSegments:
            pending = {name: column[numComplete:].copy()
                       for name, column in segments.items()}

    if pending is not None and padIncomplete:
        numPad = eventsPerBatch - len(pending["startTime"])
        padded = {}
        for name, column in pending.items():
            if name == "pathCount":
                # no paths, so the padded segments emit nothing
                fill = np.zeros((numPad, *column.shape[1:]), column.dtype)
            elif name in ("startPosition", "endPosition"):
                fill = np.zeros((numPad, *column.shape[1:]), column.dtype)
            else:
                # Repeat the last real value rather than zeroing: a zero event or
                # group id would make the last task look like it ended on event 0
                # of group 0, and a zero spectrum address would be a null table.
                fill = np.repeat(column[-1:], numPad, axis=0)
            padded[name] = np.concatenate((column, fill))
        yield make_task(padded)


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

    Tasks are emitted **in input order**: batch k holds segments
    k*eventsPerBatch .. (k+1)*eventsPerBatch-1 of the stream that results from
    walking `events` and splitting every deposition step into one segment per 
    scintillation component. All segments of a step, and hence of an event, are 
    therefore adjacent.

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
    host memory bounded. The trade-off is that `maxBatchSize` is not known up
    front, so the tracer capacity must be fixed a priori (each yielded task still
    carries its own `_batchSize`).

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
        Source to sample wavelengths from. Required for forward mode. A
        `BatchedSpectrumWavelengthSource` gives every segment its own spectrum
        and requires a deterministic segment assignment, i.e. `emitParticles` or
        `numSamples`.
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
    energyDeposition: float | Iterable[float], default=1.0 keV
        Energy deposited per segment. Will be ignored if `emitParticles` is `True`.
    photonYield: float | Iterable[float], default=10000/MeV
        Number of generated photons per deposited energy of each segment. Will be
        ignored if `emitParticles` is `True`.
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
        reconstruct the origin of a recorded detector hit. Setting numSamples leads
        to a map between thread index and original event.
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
        energyDeposition: float | Iterable[float] = 1.0 * u.keV,
        photonYield: float | Iterable[float] = 10 / u.keV,
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

        class LightParams(Structure):
            _fields_ = fields

        # the extra draw is the one importance sampling the segment; a
        # deterministic assignment needs none
        addRNG = 1 if eventsPerBatch > 1 and not deterministic else 0
        extra = {"energyDeposition", "photonYield"}
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
        energyDeposition = _norm_scalar(energyDeposition,"energyDeposition")
        mediumIdx        = _norm_scalar(mediumIdx,       "mediumIdx")
        photonYield      = _norm_scalar(photonYield,     "photonYield")
        timeConstant     = _norm_scalar(timeConstant,    "timeConstant")

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
        )
        code = loadShader("lightsource/scintillation/deposition_event/backward.glsl")
        return preamble + code
    
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
                expectedPhotons = (
                    np.asarray(self.getParam("energyDeposition"), np.float64)
                    * np.asarray(self.getParam("photonYield"), np.float64))
                self.setParam("photonsPerRay", np.divide(
                    expectedPhotons, pathCount,
                    out=np.zeros_like(expectedPhotons), where=pathCount > 0))
        else:
            photonYield = self.getParam("photonYield")
            energyDeposition = self.getParam("energyDeposition")
            # compute culmulative expected number of photons for sampling
            self.setParam("cumulativePhotonNumber", np.cumsum(
                np.asarray(energyDeposition) * np.asarray(photonYield)))
