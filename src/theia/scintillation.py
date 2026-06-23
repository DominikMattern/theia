from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from collections.abc import Iterable
from ctypes import Structure, c_float, c_uint32
from hephaistos.glsl import vec3, mat, vec
from theia.compiler import createPreamble, loadShader

from theia.light import LightSource, WavelengthSource
from theia.material import Medium, MaterialStore
import theia.units as u

__all__ = [
    "EnergyDepositionEvent",
    "ScintillationDepositionEventSource",
    "createTasksFromEvents",
]


def __dir__():
    return __all__


@dataclass
class EnergyDepositionEvent:
    """
    Data class to store energy deposition events in a form that is compatible
    with the output of remage simulations.
    """
    eventId: int
    medium: Medium
    startPosition: tuple[float, float, float]  = (0.0, 0.0, 0.0)
    endPosition: tuple[float, float, float]  = (0.0, 0.0, 0.0)
    startTime: float = 0.0
    startVelocity: float = 0.0
    endVelocity: float = 0.0
    energyDeposition: float = 1.0 * u.keV
    particleCode: int = 11


def createTasksFromEvents(events: Iterable[EnergyDepositionEvent],
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
    events: Iterable[EnergyDepositionEvent]
        List of events to generate the tasks for.
    matStore: MaterialStore
        Material Store that is used for the simulation.
    eventsPerBatch: int, default=1
        Number of track segments per task. Events are grouped into batches by
        (medium, scintillation component, particle code) — all three must match
        within a batch. The `ScintillationDepositionEventSource` must be
        created with the same batchSize.
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
    tasks = []
    accumulators = {}   # group_key -> list of pending per-event dicts
    group_info = {}     # group_key -> (mediumIdx, effectiveYield, timeConstant, spectrumAddr)
    seed_sequence = np.random.SeedSequence(RNGSeed)
    maxBatchSize = 0

    _ZERO = {
        'startPosition': (0.0, 0.0, 0.0),
        'endPosition':   (0.0, 0.0, 0.0),
        'startTime':      0.0,
        'startVelocity':  0.0,
        'endVelocity':    0.0,
        'energyDeposition': 0.0,
    }

    def _flush(group_key, batch, pad_to_full=False):
        nonlocal maxBatchSize
        medIdx, eff_yield, tc, spec = group_info[group_key]
        if pad_to_full:
            batch = batch + [_ZERO] * (eventsPerBatch - len(batch))
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
            expected_number = np.clip(np.asarray([e['energyDeposition'] for e in batch]) * eff_yield,0,None)
            resolutionscale = 1.0
            if "scintillation_resolutionscale" in event.medium.properties:
                resolutionscale = event.medium.properties["scintillation_resolutionscale"].value[1]
            number = np.round(expected_number + np.sqrt(expected_number) * resolutionscale * rdm)
            batchSize = int(np.sum(number))
            maxBatchSize = batchSize if batchSize > maxBatchSize else maxBatchSize
            task["lightSource__numPhotons"] = list(number)
            task["_batchSize"] = batchSize
        else:
            task["lightSource__energyDeposition"] = [e['energyDeposition'] for e in batch]
            task["lightSource__photonYield"] = eff_yield
        
        tasks.append(task)

    def _add(group_key, edict, medIdx, eff_yield, tc, spec):
        if group_key not in accumulators:
            accumulators[group_key] = []
            group_info[group_key] = (medIdx, eff_yield, tc, spec)
        accumulators[group_key].append(edict)
        if len(accumulators[group_key]) == eventsPerBatch:
            _flush(group_key, accumulators.pop(group_key))

    for event in events:
        if event.particleCode == 11:
            bR = event.medium.properties["scintillation_branching_ratio_electron"].value[1]
            pY = event.medium.properties["scintillation_yield_electron"].value[1]
        elif event.particleCode == 2212:
            bR = event.medium.properties["scintillation_branching_ratio_proton"].value[1]
            pY = event.medium.properties["scintillation_yield_proton"].value[1]
        elif event.particleCode == 1000020040:
            bR = event.medium.properties["scintillation_branching_ratio_alpha"].value[1]
            pY = event.medium.properties["scintillation_yield_alpha"].value[1]
        elif event.particleCode > 1000000000:
            bR = event.medium.properties["scintillation_branching_ratio_ion"].value[1]
            pY = event.medium.properties["scintillation_yield_ion"].value[1]
        else:
            print(f"Warning: Scintillation of particle with code {event.particleCode} not supported. "
                  "This scintillation step will be skipped.")
            continue

        edict = {
            'startPosition':    event.startPosition,
            'endPosition':      event.endPosition,
            'startTime':        event.startTime,
            'startVelocity':    event.startVelocity,
            'endVelocity':      event.endVelocity,
            'energyDeposition': event.energyDeposition,
        }
        medIdx = matStore.media[event.medium.name]

        tc1   = event.medium.properties["scintillation_time_constant_1"].value[1]
        spec1 = event.medium.properties["scintillation_spectrum_1_sampler"].data
        _add((event.medium.name, 1, event.particleCode), edict,
             medIdx, pY * bR, tc1, spec1)

        if bR < 1.0:
            tc2   = event.medium.properties["scintillation_time_constant_2"].value[1]
            spec2 = event.medium.properties["scintillation_spectrum_2_sampler"].data
            _add((event.medium.name, 2, event.particleCode), edict,
                 medIdx, pY * (1.0 - bR), tc2, spec2)

    for group_key, batch in accumulators.items():
        if padIncomplete:
            _flush(group_key, batch, pad_to_full=True)
        # else: discard incomplete last batch

    maxBatchSize = None if maxBatchSize == 0 else maxBatchSize

    return tasks, maxBatchSize


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
