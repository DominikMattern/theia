from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.constants as consts
from scipy.integrate import quad
from scipy.stats.sampling import NumericalInversePolynomial

import hephaistos as hp
from hephaistos.pipeline import PipelineStage, SourceCodeMixin
from hephaistos.queue import IOQueue

from ctypes import Structure, c_float, c_uint32, c_uint64, addressof, memmove, sizeof
from hephaistos.glsl import buffer_reference, uvec3, vec2, vec3
from theia.util import createCType

from theia.compiler import compileShader, createPreamble, loadShader
from theia.lookup import uploadTables
from theia.material import MaterialStore
from theia.random import RNG
from theia.ray import RayModel
import theia.units as u

from warnings import warn

from collections.abc import Callable, Iterable
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "BatchedSpectrumWavelengthSource",
    "CherenkovLightSource",
    "ConeLightSource",
    "ConstWavelengthSource",
    "FunctionWavelengthSource",
    "HostLightSource",
    "HostWavelengthSource",
    "LightSampler",
    "LightSource",
    "LightSourceRNGDraws",
    "MuonTrackLightSource",
    "ParticleCascadeLightSource",
    "PencilLightSource",
    "SpectrumWavelengthSource",
    "SphericalLightSource",
    "StreamingHostLightSource",
    "StreamingHostWavelengthSource",
    "UniformWavelengthSource",
    "VoxelLightSource",
    "WavelengthSource",
    "frankTamm",
]


def __dir__():
    return __all__


def _make_scalar_array(ctype, dtype, n):
    """
    Creates a ctypes class mirroring a GLSL array of `n` scalars.

    A single array field rather than n named ones: ctypes assembles the layout of
    a structure field by field and does so in quadratic time, which for the batch
    sizes used by scintillation sources dominates the whole setup (n = 262144
    took minutes). One array field has the exact same memory layout - n tightly
    packed scalars, which is what the shader expects with `layout(scalar)` - and
    is built in constant time.
    """

    class ScalarArray(Structure):
        _fields_ = [("items", ctype * n)]
        _dim_ = n

        # `PipelineStage` reads and writes parameters through `value` only. The
        # setter writes the whole array with a single memmove; assigning element
        # by element is far too slow for typical batch sizes.
        @property
        def value(self):
            return self.items[:]

        @value.setter
        def value(self, value):
            arr = np.ascontiguousarray(value, dtype=dtype)
            if arr.size != n:
                raise ValueError(f"expected {n} values, got {arr.size}")
            memmove(addressof(self), arr.ctypes.data, sizeof(self))

        def __len__(self):
            return n

        def __getitem__(self, idx):
            return self.items[idx]

        def __setitem__(self, idx, value):
            self.items[idx] = value

    return ScalarArray


class WavelengthSource(SourceCodeMixin):
    """
    Base class for samplers producing wavelengths.
    """

    name = "Wavelength Sampler"

    def __init__(
        self,
        *,
        nRNGSamples: int = 0,
        params: dict[str, type[Structure]] = {},
        extra: set[str] = set(),
    ) -> None:
        super().__init__(params, extra)
        self._nRNGSamples = nRNGSamples

    @property
    def nRNGSamples(self) -> int:
        """Number of RNG calls per sample"""
        return self._nRNGSamples


class HostWavelengthSource(WavelengthSource):
    """
    Wavelength source passing samples from CPU to GPU.

    Parameters
    ----------
    capacity: int
        Maximum number of samples that can be drawn per run
    emitParticle: int
        If `False`, samples have a dedicated contribution value.
    updateFn: (HostWavelengthSource, int) -> None | None, default=None
        Optional update function called before the pipeline processes a task.
        `i` is the i-th configuration the update should affect.
        Can be used to stream in new samples on demand.
    extra: set[str], default={}
        Set of extra parameter name, that can be set and retrieved using the
        stage api.
    """

    class WavelengthParams(Structure):
        _fields_ = [("_queueAdr", buffer_reference), ("_queueSize", c_uint32)]

    def __init__(
        self,
        capacity: int,
        *,
        emitParticle: bool = False,
        updateFn: Callable[[HostWavelengthSource, int], None] | None = None,
        extra: set[str] = set(),
    ) -> None:
        super().__init__(
            params={"WavelengthParams": HostWavelengthSource.WavelengthParams},
            extra=extra,
        )
        # allocate memory
        fields = [("wavelength", c_float)]
        if not emitParticle:
            fields.append(("contrib", c_float))
        item = createCType("Item", fields)
        self._queue = IOQueue(item, capacity, mode="update", skipCounter=True)
        assert self._queue.tensor is not None
        # save params
        self._capacity = capacity
        self._emitParticle = emitParticle
        self._updateFn = updateFn
        self.setParams(_queueAdr=self._queue.tensor.address, _queueSize=capacity)

    @property
    def capacity(self) -> int:
        """Maximum number of samples that can be drawn per run"""
        return self._capacity

    @property
    def emitsParticle(self) -> bool:
        """If `False`, samples have a dedicated contribution value."""
        return self._emitParticle

    @property
    def queue(self) -> IOQueue:
        """Queue containing the data for the next batch"""
        return self._queue

    @property
    def sourceCode(self) -> str:
        preamble = createPreamble(WAVELENGTH_SOURCE_EMIT_PARTICLE=self.emitsParticle)
        code = loadShader("wavelengthsource/host.glsl")
        return preamble + code

    def update(self, i: int) -> None:
        if self._updateFn is not None:
            self._updateFn(self, i)
        super().update(i)

    def run(self, i: int) -> list[hp.Command]:
        return [*self.queue.run(i), *super().run(i)]


class StreamingHostWavelengthSource(HostWavelengthSource):
    """
    Wavelength source using a provided data source to sequentially feed pre
    sampled wavelengths to the device. Subsequent batches move forward into the
    provided data streaming it to the device. Providing new data resets the
    stream.

    Parameters
    ----------
    batchSize: int
        Number of wavelengths to sample per batch
    data: dict[str, ArrayLike] | None, default=None
        Data source from which the batches are produced
    emitParticle: bool, default=False
        Whether the source should produce particles or not.

    Stage Parameters
    ----------------
    data: ArrayLike | None
        Data source from which the batches are produced.

    Note
    ----
    There is no special handling for when the last batch from the data is not
    filling the queue completely. Data in the queue beyond the end is likely old
    or garbage, but the source does not prevent the tracer from reading it.
    """

    def __init__(
        self,
        batchSize: int,
        *,
        data: dict[str, ArrayLike] | None = None,
        emitParticle: bool = False,
    ) -> None:
        super().__init__(
            batchSize,
            emitParticle=emitParticle,
            updateFn=self._updateData,
            extra={"data"},
        )
        self._dataFields = set(self.queue.view(0).fields)
        self.data = data

    @property
    def data(self) -> dict[str, NDArray] | None:
        """Data used as streaming source"""
        return self._data

    @data.setter
    def data(self, value: dict[str, ArrayLike] | None) -> None:
        self._offset = 0
        self._dataSize = 0
        if value is None:
            self._data = None
            return
        # check if all the needed fields are there
        if not self._dataFields.issubset(value.keys()):
            m = self._dataFields.difference(value.keys())
            raise ValueError(f"provided data misses expected fields: {m}")
        self._dataSize = np.size(value["wavelength"], 0)
        # ensure all fields have the same size
        for field in self._dataFields:
            if np.size(value[field], 0) != self._dataSize:
                raise ValueError("fields of provided data differ in size!")
            # TODO: Check correct dimensions, too?
        # create ndarray and store them for later
        self._data = {f: np.asarray(value[f]) for f in self._dataFields}

    @data.deleter
    def data(self) -> None:
        self.data = None

    @staticmethod
    def _updateData(source: HostWavelengthSource, i: int) -> None:
        assert isinstance(source, StreamingHostWavelengthSource)
        if source.data is None:
            warn(
                "StreamingHostWavelengthSource: "
                "New data was requested but no source was given!"
            )
            return
        if source._offset >= source._dataSize:
            warn("StreamingHostWavelengthSource ran out of data!")
            return

        # calculate next data range
        start = source._offset
        end = min(source._offset + source.capacity, source._dataSize)
        count = end - start
        source._offset = end
        # copy new data block
        view = source.queue.view(i)
        for field in source._dataFields:
            view[field][:count] = source.data[field][start:end]


class ConstWavelengthSource(WavelengthSource):
    """
    Sampler generating photons of constant wavelength.

    Parameters
    ----------
    wavelength: float, default=600.0nm
        Constant wavelength

    Stage Parameters
    ----------------
    wavelength: float, default=600.0nm
        Constant wavelength
    """

    name = "Const Wavelength Source"

    class WavelengthParams(Structure):
        _fields_ = [("wavelength", c_float)]

    def __init__(self, wavelength: float = 600.0 * u.nm) -> None:
        super().__init__(params={"WavelengthParams": self.WavelengthParams})
        self.setParams(wavelength=wavelength)

    @property
    def sourceCode(self) -> str:
        return loadShader("wavelengthsource/const.glsl")


class SpectrumWavelengthSource(WavelengthSource):
    """
    Samples wavelengths based on the given spectrum.

    Parameters
    ----------
    spectrumTableAddress: int
        GPU address of a table containing the percent point function of the
        light spectrum.

    Stage Parameters
    ----------------
    spectrumTableAddress: int
        GPU address of a table containing the percent point function of the
        light spectrum.
    """

    name = "Spectrum Wavelength Source"

    class WavelengthParams(Structure):
        _fields_ = [("spectrumTableAddress", c_uint64)]

    def __init__(self, spectrumTableAddress: int) -> None:
        super().__init__(
            nRNGSamples=1,
            params={"WavelengthParams": self.WavelengthParams},
        )
        self.setParams(spectrumTableAddress=spectrumTableAddress)

    @property
    def sourceCode(self) -> str:
        return loadShader("wavelengthsource/spectrum.glsl")


class BatchedSpectrumWavelengthSource(WavelengthSource):
    """
    Samples wavelengths from a per batch element spectrum.

    Parameters
    ----------
    spectraPerBatch: int
        Number of batch elements a dispatch is partitioned into. Must match the
        light source's.
    spectrumTableAddress: int | Iterable[int], default=0
        GPU address of the percent point function of each element's spectrum. A
        single address is broadcast to every element; an array must have length
        `spectraPerBatch`.
    count: int | Iterable[int], default=0
        Number of paths each element owns, broadcast like
        `spectrumTableAddress`.
    axisValue: float | Iterable[float] | None, default=None
        Enables a second table axis. `None` leaves the source exactly as it was:
        no field, and every address is read as a `Table1D`. Otherwise each
        element carries a coordinate on the second axis, and **NaN selects the
        1D lookup** for that element - so 1D and 2D spectra may be mixed freely
        within one batch. Defaults to NaN, i.e. 1D everywhere.

    Stage Parameters
    ----------------
    spectrumTableAddress: int | Iterable[int]
        GPU address of the percent point function of each element's spectrum.
    count: int | Iterable[int]
        Number of paths each element owns.
    axisValue: float | Iterable[float]
        Coordinate on the second axis; NaN selects a 1D table. Only present if
        the source was constructed with `axisValue`.

    Note
    ----
    In a 2D table the second axis is the **slow** one (`u`) and the sampled
    quantile the **fast** one (`v`), matching the row-major layout of `Table2D`.
    All paths of one batch element then read from the same two rows, which keeps
    the lookup cache friendly.
    """

    name = "Batched Spectrum Wavelength Source"

    def __init__(
        self,
        spectraPerBatch: int,
        *,
        spectrumTableAddress: int | Iterable[int] = 0,
        count: int | Iterable[int] = 0,
        axisValue: float | Iterable[float] | None = None,
    ) -> None:
        if spectraPerBatch < 1:
            raise ValueError(
                f"spectraPerBatch must be at least one, got {spectraPerBatch}")

        # the field is added only when asked for, so a source without a second
        # axis keeps its old memory layout and compiles a byte identical shader
        fields = [
            ("spectrumTableAddress",
             _make_scalar_array(c_uint64, np.uint64, spectraPerBatch)),
            ("cumulativeCount",
             _make_scalar_array(c_uint32, np.uint32, spectraPerBatch)),
        ]
        if axisValue is not None:
            fields.append(
                ("axisValue",
                 _make_scalar_array(c_float, np.float32, spectraPerBatch)))

        class WavelengthParams(Structure):
            _fields_ = fields

        super().__init__(
            nRNGSamples=1,
            params={"WavelengthParams": WavelengthParams},
            extra={"count"},
        )
        self._spectraPerBatch = spectraPerBatch
        self._hasSecondAxis = axisValue is not None

        def norm(value, name, dtype):
            """Broadcast a scalar over the batch, or check an array's length."""
            array = np.asarray(value, dtype)
            if array.ndim == 0:
                return np.full(spectraPerBatch, array, dtype)
            if array.size != spectraPerBatch:
                raise ValueError(f"'{name}' must have length {spectraPerBatch}, "
                                 f"got {array.size}")
            return array

        params = {
            "spectrumTableAddress": norm(spectrumTableAddress,
                                         "spectrumTableAddress", np.uint64),
            "count": norm(count, "count", np.uint32),
        }
        if self._hasSecondAxis:
            params["axisValue"] = norm(axisValue, "axisValue", np.float32)
        self.setParams(**params)

    @property
    def hasSecondAxis(self) -> bool:
        """Whether elements carry a coordinate on a second table axis"""
        return self._hasSecondAxis

    @property
    def spectraPerBatch(self) -> int:
        """Number of batch elements a dispatch is partitioned into"""
        return self._spectraPerBatch

    @property
    def sourceCode(self) -> str:
        preamble = createPreamble(
            WAVELENGTH_SOURCE_SPECTRA_PER_BATCH=self._spectraPerBatch,
            WAVELENGTH_SOURCE_SECOND_AXIS=self._hasSecondAxis,
        )
        return preamble + loadShader("wavelengthsource/spectrum_batched.glsl")

    def _finishParams(self, i: int) -> None:
        super()._finishParams(i)
        # accumulate in 64 bit: the running sum may exceed what a single element
        # can hold long before the total exceeds the uint32 the shader reads
        c = np.cumsum(np.asarray(self.getParam("count"), np.uint64))
        if len(c) and c[-1] > np.iinfo(np.uint32).max:
            raise ValueError(
                f"{c[-1]} paths exceed the {np.iinfo(np.uint32).max} the shader "
                "can index; use a smaller batch")
        self.setParam("cumulativeCount", c)


class UniformWavelengthSource(WavelengthSource):
    """
    Sampler generating photons uniform in wavelength and time.

    Parameters
    ----------
    lambdaRange: (float, float), default=(300.0, 700.0)
        min and max wavelength the source emits
    normalize: bool, default=True
        If True, adds an extra factor of one over the lambdaRange to the
        contribution to normalize the light source.

    Stage Parameters
    ----------------
    lambdaRange: (float, float), default=(300.0, 700.0)
        min and max wavelength the source emits
    normalize: bool, default=True
        If True, sets the contribution per sampled wavelength to 1.0. Normally,
        this would be the inverse sample probability, which here equals to the
        lambda range.

    Note
    ----
    In combination with light sources that allow settings the total amount of
    photons per wavelength via e.g. a parameter `budget`, setting `normalize` to
    `True` will make `budget` the total amount of photons regardless of
    `lambdaRange`.
    """

    class SourceParams(Structure):
        _fields_ = [
            ("lambdaRange", vec2),
            ("_contrib", c_float),
        ]

    def __init__(
        self,
        *,
        lambdaRange: tuple[float, float] = (300.0, 700.0),
        normalize: bool = True,
    ) -> None:
        super().__init__(
            nRNGSamples=1,
            params={"WavelengthParams": self.SourceParams},
            extra={"normalize"},
        )
        # save params
        self.setParams(lambdaRange=lambdaRange, normalize=normalize)

    @property
    def normalize(self) -> bool:
        """Whether to normalize this source."""
        return self._normalize

    @normalize.setter
    def normalize(self, value: bool) -> None:
        self._normalize = value

    @property
    def sourceCode(self) -> str:
        return loadShader("wavelengthsource/uniform.glsl")

    def _finishParams(self, i: int) -> None:
        c = 1.0
        lr = self.getParam("lambdaRange")
        lr = lr[1] - lr[0]
        if lr != 0.0 and not self.normalize:
            c *= abs(lr)
        self.setParam("_contrib", c)


class FunctionWavelengthSource(WavelengthSource):
    """
    Wavelength source generating samples by importance sampling the given
    function or distribution. Numerically integrates and inverts the
    distribution before discretizing the result into a look up table used by
    the GPU.

    Parameters
    ----------
    fn: (float) -> float
        Function to be importance sampled, mapping wavelengths in nm to a
        scalar.
    lambdaRange: (float, float), default=(300.0, 700.0)nm
        Wavelength range of the generated samples.
    numSamples: int, default=1024
        Number of entries in the final look up table.
    """

    name = "Function Wavelength Source"

    class WavelengthParams(Structure):
        _fields_ = [("_table", c_uint64), ("_contrib", c_float)]

    def __init__(
        self,
        fn: Callable[[float], float],
        *,
        lambdaRange: tuple[float, float] = (300.0, 700.0) * u.nm,
        numSamples: int = 1024,
    ) -> None:
        super().__init__(
            nRNGSamples=1,
            params={"WavelengthParams": self.WavelengthParams},
        )
        self._updateFn(fn, lambdaRange, numSamples)

    @property
    def sourceCode(self) -> str:
        return loadShader("wavelengthsource/function.glsl")

    def _updateFn(
        self,
        fn: Callable[[float], float],
        lambdaRange: tuple[float, float],
        numSamples: int,
    ) -> None:
        # integrate fn to get constant contribution
        contrib, _ = quad(fn, *lambdaRange)

        class Dist:
            """Bundle fn into a class for use with scipy"""

            def pdf(self, x):
                return fn(x)

        # invert cdf
        inv_cdf = NumericalInversePolynomial(Dist(), domain=lambdaRange)
        # sample inverted cdf
        u = np.linspace(0.0, 1.0, numSamples)
        x = inv_cdf.ppf(u)
        # create table from samples
        self._tableMemory, [table_adr] = uploadTables([x])

        # update params
        self.setParams(_table=table_adr, _contrib=contrib)


@dataclass
class LightSourceRNGDraws:
    """Number of RNG calls in each light source function"""

    sampleLightForward: int = 0
    sampleLightBackward: int = 0
    sampleLightNeeStepSize: int = 0
    sampleLightMis: int = 0


class LightSource(PipelineStage):
    """Base class for samplers producing light rays"""

    name = "Light Source"

    def __init__(
        self,
        wavelengthSource: WavelengthSource | None = None,
        *,
        rngDraws: LightSourceRNGDraws,
        supportsMis: bool = False,
        params: dict[str, type[Structure]] = {},
        extra: set[str] = set(),
    ) -> None:
        super().__init__(params, extra)
        self.rngDraws = rngDraws
        self._supportsMis = supportsMis
        self._wavelengthSource = wavelengthSource

    @property
    def backwardSourceCode(self) -> str | None:
        """
        Source code implementing light source sampling in backward tracing.
        `None` if backward tracing is not supported.
        """
        return None

    @property
    def forwardSourceCode(self) -> str | None:
        """
        Source code implementing light source sampling in forward tracing.
        `None` if forward tracing is not supported.
        """
        return None

    @property
    def supportsMis(self) -> bool:
        """
        Whether the light source provides methods required for MIS weighted NEE
        in backward mode.
        """
        return self._supportsMis

    @property
    def wavelengthSource(self) -> WavelengthSource | None:
        """Optional wavelength source used for sampling wavelengths"""
        return self._wavelengthSource

    def run(self, i: int) -> list[hp.Command]:
        # most light sources will not issue any commands
        # so an empty list is a sensible default
        return []


class LightSampler(PipelineStage):
    """
    Samples a given light source and puts the result into a queue for processing
    on the CPU.

    Parameters
    ----------
    batchSize: int
        Number of samples to draw per batch.
    source: LightSource
        Light source to sample from.
    ray: RayModel
        Model describing the structure of rays to sample.
    rng: RNG | None, default=None
        Optional RNG the light source may require.
    materials: MaterialStore | None, default=None
        Optional material store the light source or ray model may require.
    """

    name = "Light Sampler"

    class SampleParams(Structure):
        _fields_ = [
            ("_batchSize", c_uint32),
            ("baseCount", c_uint32),
            ("_queueAdr", buffer_reference),
            ("_queueSize", c_uint32),
        ]

    def __init__(
        self,
        batchSize: int,
        source: LightSource,
        ray: RayModel,
        *,
        rng: RNG | None = None,
        materials: MaterialStore | None = None,
    ) -> None:
        # check if we have RNG if needed
        if source.forwardSourceCode is None:
            raise ValueError("Light source does not support forward mode")
        if source.rngDraws.sampleLightForward > 0 and rng is None:
            raise ValueError("Light source requires a RNG but none was given")
        super().__init__({"SampleParams": LightSampler.SampleParams})
        # create queue
        self._queue = IOQueue(
            ray.forwardItem, batchSize, mode="retrieve", skipCounter=True
        )
        assert self._queue.tensor is not None
        # create empty material store if needed
        if materials is None:
            materials = MaterialStore([], mediaSlots=ray.requiredMediumProperties)
        # save params
        self._source = source
        self._ray = ray
        self._rng = rng
        self._materials = materials
        self.setParams(
            _batchSize=batchSize,
            baseCount=0,
            _queueAdr=self._queue.tensor.address,
            _queueSize=batchSize,
        )

        # create code
        lamSource = source.wavelengthSource
        headers = {
            "light.glsl": source.forwardSourceCode,
            "photon.glsl": "" if lamSource is None else lamSource.sourceCode,
            "ray.glsl": ray.sourceCode,
            "rng.glsl": "" if rng is None else rng.sourceCode,
            **materials.header,
        }
        code = compileShader("lightsource/sample/forward.glsl", "", headers)
        self._program = hp.Program(code)
        # bind memory
        materials.bindParams(self._program)

    @property
    def batchSize(self) -> int:
        """Number of samples drawn per run"""
        return self._batchSize

    @property
    def materials(self) -> MaterialStore | None:
        """Optional material store in use"""
        return self._materials

    @property
    def queue(self) -> IOQueue:
        """Queue holding the sampled light rays"""
        return self._queue

    @property
    def rayModel(self) -> RayModel:
        """Ray model used for sampling"""
        return self._ray

    @property
    def rng(self) -> RNG | None:
        """The random number generator used for sampling. None, if none used."""
        return self._rng

    @property
    def source(self) -> LightSource:
        """Light source that is sampled"""
        return self._source

    def collectStages(self) -> list[PipelineStage]:
        """
        Returns a list of all stages involved with this sampler in the correct
        order suitable for creating a pipeline.
        """
        stages: list[PipelineStage] = []
        if self.rng is not None:
            stages.append(self.rng)
        if self.source.wavelengthSource is not None:
            stages.append(self.source.wavelengthSource)
        stages.extend([self.source, self])
        return stages

    def run(self, i: int) -> list[hp.Command]:
        for stage in self.collectStages():
            stage.bindParams(self._program, i)
        groups = -(self.batchSize // -512)
        return [
            self._program.dispatch(groups),
            *self._queue.run(i),
        ]


class HostLightSource(LightSource):
    """
    Light source passing samples from the CPU to the GPU

    Parameters
    ----------
    capacity: int
        Maximum number of samples that can be drawn per run
    ray: RayModel | type[Structure]:
        Model describing the structure of rays in the queue
    updateFn: (HostLightSource, int) -> None | None, default=None
        Optional update function called before the pipeline processes a task.
        `i` is the i-th configuration the update should affect.
        Can be used to stream in new samples on demand.
    extra: set[str], default={}
        Set of extra parameter name, that can be set and retrieved using the
        stage api.
    """

    name = "Host Light Source"

    class LightParams(Structure):
        _fields_ = [("_queueAdr", buffer_reference), ("_queueSize", c_uint32)]

    def __init__(
        self,
        capacity: int,
        ray: RayModel | type[Structure],
        *,
        updateFn: Callable[[HostLightSource, int], None] | None = None,
        extra: set[str] = set(),
    ) -> None:
        super().__init__(
            rngDraws=LightSourceRNGDraws(),
            params={"LightParams": HostLightSource.LightParams},
            extra=extra,
        )

        if isinstance(ray, RayModel):
            ray = ray.forwardItem
        # allocate queue
        self._queue = IOQueue(ray, capacity, mode="update", skipCounter=True)
        assert self._queue.tensor is not None
        # save params
        self._capacity = capacity
        self._updateFn = updateFn
        self.setParams(_queueAdr=self._queue.tensor.address, _queueSize=capacity)

    @property
    def queue(self) -> IOQueue:
        """Queue holding the samples for the next batch"""
        return self._queue

    @property
    def forwardSourceCode(self) -> str:
        return loadShader("lightsource/host.glsl")

    def update(self, i: int) -> None:
        if self._updateFn is not None:
            self._updateFn(self, i)
        super().update(i)

    def run(self, i: int) -> list[hp.Command]:
        return [*self.queue.run(i), *super().run(i)]


class StreamingHostLightSource(HostLightSource):
    """
    Light source using a provided data source to sequentially feed light rays to
    the device. Subsequent batches move forward into the provided data streaming
    it to the device. Providing new data resets the stream.

    Parameters
    ----------
    capacity: int
        Maximum number of rays that can be drawn per run
    ray: RayModel | type[Structure]
        Ray model or structure describing the data in the queue to be fed to the
        GPU
    data: dict[str, ArrayLike] | None, default=None
        Data source from which the batches are produced.

    Stage Parameters
    ----------------
    data: dict[str, ArrayLike] | None
        Data source from which the batches are produced.

    Note
    ----
    There is no special handling for when the last batch from the data is not
    filling the queue completely. Data in the queue beyond the end is likely old
    or garbage, but the source does not prevent the tracer from reading it.
    """

    def __init__(
        self,
        capacity: int,
        ray: RayModel | type[Structure],
        *,
        data: dict[str, ArrayLike] | None = None,
    ) -> None:
        super().__init__(capacity, ray, updateFn=self._updateData, extra={"data"})
        self._dataFields = set(self.queue.view(0).fields)
        self.data = data

    @property
    def data(self) -> dict[str, NDArray] | None:
        """Data used as streaming source"""
        return self._data

    @data.setter
    def data(self, value: dict[str, ArrayLike] | None) -> None:
        self._offset = 0
        self._dataSize = 0
        if value is None:
            self._data = None
            return
        # check if all the needed fields are there
        if not self._dataFields.issubset(value.keys()):
            m = self._dataFields.difference(value.keys())
            raise ValueError(f"provided data misses expected fields: {m}")
        self._dataSize = np.size(value["wavelength"], 0)
        # ensure all fields have the same size
        for field in self._dataFields:
            if np.size(value[field], 0) != self._dataSize:
                raise ValueError("fields of provided data differ in size!")
            # TODO: Check correct dimensions, too?
        # create ndarray and store them for later
        self._data = {f: np.asarray(value[f]) for f in self._dataFields}

    @data.deleter
    def data(self) -> None:
        self.data = None

    @staticmethod
    def _updateData(source: HostLightSource, i: int) -> None:
        assert isinstance(source, StreamingHostLightSource)
        if source.data is None:
            warn(
                "StreamingHostLightSource: "
                "New data was requested but no source was given!"
            )
            return
        if source._offset >= source._dataSize:
            warn("StreamingHostLightSource ran out of data!")
            return

        # calculate next data range
        start = source._offset
        end = min(source._offset + source._capacity, source._dataSize)
        count = end - start
        source._offset = end
        # copy new data block
        view = source.queue.view(i)
        for field in source._dataFields:
            view[field][:count] = source.data[field][start:end]


class ConeLightSource(LightSource):
    """
    Point light source radiating light in a specified cone.
    Can optionally be constant polarized.

    Parameters
    ----------
    wavelengthSource: WavelengthSource
        Source to sample wavelengths from. Required for forward mode.
    mediumIdx: int
        Index of the medium the light source is located in
    position: (float, float, float), default=(0.0, 0.0, 0.0)
        Position of the light source.
    direction: (float, float, float), default(1.0, 0.0, 0.0)
        Direction of the opening cone.
    timeRange: (float, float), default=(0.0, 100.0)
        start and stop time of the light source
    cosOpeningAngle: float, default=0.5
        Cosine of the cones opening angle
    budget: float, default=1.0
        Total amount of energy or photons the light source distributes among the
        sampled photons per wavelength. Ignored if source emits particles.
    emitParticles: bool, default=False
        Whether to emit particles instead of rays. If so, `budget` will be
        ignored.
    """

    name = "Cone Light Source"

    class LightParams(Structure):
        _fields_ = [
            ("direction", vec3),
            ("cosOpeningAngle", c_float),
            ("position", vec3),
            ("mediumIdx", c_uint32),
            ("_contribFwd", c_float),
            ("_contribBwd", c_float),
            ("timeRange", vec2),
        ]

    def __init__(
        self,
        wavelengthSource: WavelengthSource | None = None,
        *,
        mediumIdx: int,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        direction: tuple[float, float, float] = (1.0, 0.0, 0.0),
        timeRange: tuple[float, float] = (0.0, 100.0) * u.ns,
        cosOpeningAngle: float = 0.5,
        budget: float = 1.0,
        emitParticles: bool = False,
    ) -> None:
        lam = wavelengthSource
        draws = LightSourceRNGDraws(
            sampleLightForward=0 if lam is None else lam.nRNGSamples + 2,
            sampleLightBackward=1,
        )
        super().__init__(
            wavelengthSource,
            rngDraws=draws,
            # TODO: Implement backward MIS
            params={"LightParams": ConeLightSource.LightParams},
            extra={"budget"},
        )
        if emitParticles and budget != 1.0:
            warn("Cannot set budget: Light source emits particles")
        # save params
        self._emitParticles = emitParticles
        self._budget = budget
        self.setParams(
            position=position,
            direction=direction,
            timeRange=timeRange,
            cosOpeningAngle=cosOpeningAngle,
            mediumIdx=mediumIdx,
        )

    @property
    def budget(self) -> float:
        """
        Total energy or amount of photons the light source distributes among all
        sampled rays per wavelength. Ignored if source emits particles.
        """
        return self._budget

    @budget.setter
    def budget(self, value: float) -> None:
        if self.emitsParticles:
            warn("Cannot set budget: Light source emits particles")
        self._budget = value

    @property
    def emitsParticles(self) -> bool:
        """Whether this light source emits particles instead of light rays"""
        return self._emitParticles

    @property
    def backwardSourceCode(self) -> str | None:
        if self.emitsParticles:
            return None
        else:
            return loadShader("lightsource/cone/backward.glsl")

    @property
    def forwardSourceCode(self) -> str | None:
        if self.wavelengthSource is None:
            return None
        else:
            preamble = createPreamble(LIGHT_SOURCE_EMIT_PARTICLE=self.emitsParticles)
            code = loadShader("lightsource/cone/forward.glsl")
            return preamble + code

    def _finishParams(self, i: int) -> None:
        super()._finishParams(i)
        c = self.budget
        self.setParam("_contribFwd", c)
        # in forward parameter volume cancels with the probability
        # This is not the case in backward mode
        # -> divide contrib by parameter space volume
        c /= 2.0 * np.pi * (1.0 - self.cosOpeningAngle)
        self.setParam("_contribBwd", c)


class PencilLightSource(LightSource):
    """
    Light source generating a pencil beam

    Parameters
    ----------
    wavelengthSource: WavelengthSource
        Source to sample wavelengths from
    mediumIdx: int
        Index of the medium the light source is located in
    position: (float, float, float), default=(0.0, 0.0, 0.0)
        Start point of the ray
    direction: (float, float, float), default=(0.0, 0.0, 1.0)
        Direction of the ray
    timeRange: (float, float), default=(0.0, 100.0)
        start and stop time of the light source
    budget: float, default=1.0
        Total amount of energy or photons the light source distributes among the
        sampled photons per wavelength. Ignored if source emits particles.
    emitParticles: bool, default=False
        Whether to emit particles instead of rays. If so, `budget` will be
        ignored.
    """

    name = "Pencil Light Source"

    class LightParams(Structure):
        _fields_ = [
            ("position", vec3),
            ("direction", vec3),
            ("_budget", c_float),
            ("mediumIdx", c_uint32),
            ("timeRange", vec2),
        ]

    def __init__(
        self,
        wavelengthSource: WavelengthSource,
        *,
        mediumIdx: int,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
        timeRange: tuple[float, float] = (0.0, 100.0) * u.ns,
        budget: float = 1.0,
        emitParticles: bool = False,
    ) -> None:
        draws = LightSourceRNGDraws(
            sampleLightForward=wavelengthSource.nRNGSamples + 1,
        )
        super().__init__(
            wavelengthSource,
            rngDraws=draws,
            params={"LightParams": PencilLightSource.LightParams},
            extra={"budget"},
        )
        if emitParticles and budget != 1.0:
            warn("Cannot set budget: Light source emits particles")
        # save params
        self._emitParticles = emitParticles
        self._budget = budget
        self.setParams(
            position=position,
            direction=direction,
            timeRange=timeRange,
            mediumIdx=mediumIdx,
        )

    @property
    def budget(self) -> float:
        """
        Total energy or amount of photons the light source distributes among all
        sampled rays per wavelength. Ignored if source emits particles.
        """
        return self._budget

    @budget.setter
    def budget(self, value: float) -> None:
        if self.emitsParticles:
            warn("Cannot set budget: Light source emits particles")
        self._budget = value

    @property
    def emitsParticles(self) -> bool:
        """Whether this light source emits particles instead of light rays"""
        return self._emitParticles

    @property
    def forwardSourceCode(self) -> str:
        preamble = createPreamble(LIGHT_SOURCE_EMIT_PARTICLE=self.emitsParticles)
        code = loadShader("lightsource/pencil.glsl")
        return preamble + code


class SphericalLightSource(LightSource):
    """
    Isotropic point light source radiating light in all directions uniformly.

    Parameters
    ----------
    wavelengthSource: WavelengthSource
        Source to sample wavelengths from. Required for forward mode.
    mediumIdx: int
        Index of the medium the light source is located in
    position: (float, float, float), default=(0.0, 0.0, 0.0)
        Position of the light source.
    timeRange: (float, float), default=(0.0, 100.0)
        start and stop time of the light source
    budget: float, default=1.0
        Total amount of energy or photons the light source distributes among the
        sampled photons per wavelength. Ignored if source emits particles.
    emitParticles: bool, default=False
        Whether to emit particles instead of rays. If so, `budget` will be
        ignored.
    softeningRadius: float, default=1mm
        In backward tracing, to avoid infinities 1/r^2 gets softened for radii
        below softeningRadius.
    """

    class LightParams(Structure):
        _fields_ = [
            ("position", vec3),
            ("mediumIdx", c_uint32),
            ("timeRange", vec2),
            ("_contribFwd", c_float),
            ("_contribBwd", c_float),
            ("softeningRadius", c_float),
        ]

    def __init__(
        self,
        wavelengthSource: WavelengthSource | None = None,
        *,
        mediumIdx: int,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        timeRange: tuple[float, float] = (0.0, 100.0) * u.ns,
        budget: float = 1.0,
        softeningRadius: float = 1.0 * u.mm,
        emitParticles: bool = False,
    ) -> None:
        lam = wavelengthSource
        draws = LightSourceRNGDraws(
            sampleLightForward=0 if lam is None else lam.nRNGSamples + 2,
            sampleLightBackward=1,
            sampleLightNeeStepSize=1,
            sampleLightMis=1,
        )
        super().__init__(
            wavelengthSource,
            rngDraws=draws,
            supportsMis=True,
            params={"LightParams": SphericalLightSource.LightParams},
            extra={"budget"},
        )
        if emitParticles and budget != 1.0:
            warn("Cannot set budget: Light source emits particles")
        # save params
        self._emitParticles = emitParticles
        self._budget = budget
        self.setParams(
            position=position,
            mediumIdx=mediumIdx,
            timeRange=timeRange,
            softeningRadius=softeningRadius,
        )

    @property
    def budget(self) -> float:
        """
        Total energy or amount of photons the light source distributes among all
        sampled rays per wavelength. Ignored if source emits particles.
        """
        return self._budget

    @budget.setter
    def budget(self, value: float) -> None:
        if self.emitsParticles:
            warn("Cannot set budget: Light source emits particles")
        self._budget = value

    @property
    def emitsParticles(self) -> bool:
        """Whether this light source emits particles instead of light rays"""
        return self._emitParticles

    @property
    def forwardSourceCode(self) -> str | None:
        if self.wavelengthSource is None:
            return None
        else:
            preamble = createPreamble(LIGHT_SOURCE_EMIT_PARTICLE=self.emitsParticles)
            code = loadShader("lightsource/spherical/forward.glsl")
            return preamble + code

    @property
    def backwardSourceCode(self) -> str | None:
        if self.emitsParticles:
            return None
        else:
            return loadShader("lightsource/spherical/backward.glsl")

    def _finishParams(self, i: int) -> None:
        super()._finishParams(i)
        c = self.budget
        self.setParam("_contribFwd", c)
        # in forward parameter volume cancels with the probability
        # This is not the case in backward mode
        # -> divide contrib by parameter space volume
        c /= 4.0 * np.pi
        self.setParam("_contribBwd", c)


class VoxelLightSource(LightSource):
    """
    Isotropic light source sampling positions uniformly inside the cells (voxels)
    of a regular grid spanning an axis aligned box.

    Consecutive blocks of `raysPerCell` rays are assigned to consecutive cells,
    i.e. the ray with index `idx` within a batch starts inside the cell with
    flat index `cellOffset + idx // raysPerCell`. Cells are enumerated in C
    order, i.e. `numpy.ravel_multi_index((ix, iy, iz), gridShape)`. It is up to
    the user to ensure the batch does not run past the last cell. With the
    default grid shape of `(1, 1, 1)`, the whole box is sampled uniformly.

    If `checkMedium` is set, the source restricts the sampled positions to the
    medium it emits into by querying the scene at each candidate position. If
    none of the `maxAttempts` candidates lies inside the medium, the sample is
    rejected and the tracer does not create a ray. The number of created rays
    per cell can be retrieved via `rayCount`. Positions from which the query
    hits no surface at all are rejected as well, unless `acceptLostRays` is set:
    theia scenes have no medium of their own, so a medium extending to the
    scene's bounding box without being enclosed by a surface (e.g. the world
    volume of a GDML) can only be identified this way. The direction of the
    query is reused as emission direction. This requires the medium to only
    depend on the position, i.e. the geometry must either be closed, or be
    surrounded by a single medium and `acceptLostRays` be set. Otherwise the
    rejection depends on the direction, e.g. through an opening of the geometry,
    and biases the emission. Checking the medium is only
    supported by scene forward tracers.

    Note
    ----
    Be careful when using `checkMedium` in combination with a `MultiScene`.
    The medium check is performed against the main scene, so rays starting in
    the volume of sub-scenes will use one of the enclosing portals for the
    medium check. Please make sure that none of the portals has the given
    mediumIdx on the surface that is facing the sub-scene.

    Parameters
    ----------
    wavelengthSource: WavelengthSource
        Source to sample wavelengths from.
    mediumIdx: int
        Index of the medium the light source emits into.
    lower: (float, float, float), default=(0.0, 0.0, 0.0)
        Lower corner of the box.
    upper: (float, float, float), default=(1.0, 1.0, 1.0)
        Upper corner of the box.
    gridShape: (int, int, int), default=(1, 1, 1)
        Number of cells along each axis.
    raysPerCell: int, default=1
        Number of consecutive rays sampled in each cell.
    cellOffset: int, default=0
        Flat index of the cell the first ray of a batch is sampled in.
    autoAdvance: int | None, default=None
        Amount `cellOffset` gets incremented with each update call, i.e. after
        each batch. The first batch uses `cellOffset` as given. `None` disables
        the advance.
    timeRange: (float, float), default=(0.0, 100.0)
        Start and stop time of the light source.
    budget: float, default=1.0
        Total amount of energy or photons the light source distributes among the
        sampled photons per wavelength. Ignored if source emits particles.
    emitParticles: bool, default=False
        Whether to emit particles instead of rays. If so, `budget` will be
        ignored.
    checkMedium: bool, default=False
        Whether to only create rays at positions inside the medium given by
        `mediumIdx`.
    maxAttempts: int, default=1
        Max number of positions tried per sample if `checkMedium` is set.
    acceptLostRays: bool, default=False
        Whether to accept positions from which the medium query hits no surface
        if `checkMedium` is set, i.e. the sampled ray from this position leaves 
        the bounding box.
    maxCellsPerBatch: int | None, default=None
        Max number of cells a single batch covers, i.e. the size of the buffer
        counting the created rays. Required if `checkMedium` is set.
    """

    name = "Voxel Light Source"

    class LightParams(Structure):
        _fields_ = [
            ("lower", vec3),
            ("mediumIdx", c_uint32),
            ("_cellSize", vec3),
            ("raysPerCell", c_uint32),
            ("gridShape", uvec3),
            ("cellOffset", c_uint32),
            ("_countAdr", buffer_reference),
            ("timeRange", vec2),
            ("_contribFwd", c_float),
        ]

    def __init__(
        self,
        wavelengthSource: WavelengthSource,
        *,
        mediumIdx: int,
        lower: tuple[float, float, float] = (0.0, 0.0, 0.0),
        upper: tuple[float, float, float] = (1.0, 1.0, 1.0),
        gridShape: tuple[int, int, int] = (1, 1, 1),
        raysPerCell: int = 1,
        cellOffset: int = 0,
        autoAdvance: int | None = None,
        timeRange: tuple[float, float] = (0.0, 100.0) * u.ns,
        budget: float = 1.0,
        emitParticles: bool = False,
        checkMedium: bool = False,
        maxAttempts: int = 1,
        acceptLostRays: bool = False,
        maxCellsPerBatch: int | None = None,
    ) -> None:
        if maxAttempts < 1:
            raise ValueError("maxAttempts must be at least one")
        if raysPerCell < 1:
            raise ValueError("raysPerCell must be at least one")
        if checkMedium and maxCellsPerBatch is None:
            raise ValueError("maxCellsPerBatch is required if checkMedium is set")
        # position (3) and direction (2) per attempt + time
        nRNG = 5 * maxAttempts + 1 if checkMedium else 6
        super().__init__(
            nRNGForward=wavelengthSource.nRNGSamples + nRNG,
            wavelengthSource=wavelengthSource,
            params={"LightParams": VoxelLightSource.LightParams},
            extra={"upper", "budget", "autoAdvance"},
        )
        if emitParticles and budget != 1.0:
            warn("Cannot set budget: Light source emits particles")
        # save params
        self._emitParticles = emitParticles
        self._budget = budget
        self._upper = upper
        self._autoAdvance = autoAdvance
        self._checkMedium = checkMedium
        self._maxAttempts = maxAttempts
        self._acceptLostRays = acceptLostRays
        self._maxCells = maxCellsPerBatch
        # one counter buffer per configuration, mapped so it can be read directly
        if checkMedium:
            assert maxCellsPerBatch is not None
            self._counts = [
                hp.UnsignedIntTensor(maxCellsPerBatch, mapped=True) for _ in range(2)
            ]
            if not all(t.isMapped for t in self._counts):
                raise RuntimeError("Could not create mapped tensor")
        else:
            self._counts = []
        self.setParams(
            lower=lower,
            mediumIdx=mediumIdx,
            gridShape=gridShape,
            raysPerCell=raysPerCell,
            cellOffset=cellOffset,
            timeRange=timeRange,
        )

    @property
    def autoAdvance(self) -> int | None:
        """
        Amount `cellOffset` gets incremented after each batch or `None` if
        disabled.
        """
        return self._autoAdvance

    @autoAdvance.setter
    def autoAdvance(self, value: int | None) -> None:
        self._autoAdvance = value

    @property
    def budget(self) -> float:
        """
        Total energy or amount of photons the light source distributes among all
        sampled rays per wavelength. Ignored if source emits particles.
        """
        return self._budget

    @budget.setter
    def budget(self, value: float) -> None:
        if self.emitsParticles:
            warn("Cannot set budget: Light source emits particles")
        self._budget = value

    @property
    def cellCount(self) -> int:
        """Total number of cells in the grid"""
        return int(np.prod(self.getParam("gridShape")))

    @property
    def checkMedium(self) -> bool:
        """Whether only positions inside the medium create rays"""
        return self._checkMedium

    @property
    def emitsParticles(self) -> bool:
        """Whether this light source emits particles instead of light rays"""
        return self._emitParticles

    @property
    def maxAttempts(self) -> int:
        """Max number of positions tried per sample if the medium is checked"""
        return self._maxAttempts

    @property
    def acceptLostRays(self) -> bool:
        """
        Whether positions from which the medium query hits no surface are
        accepted
        """
        return self._acceptLostRays

    @property
    def maxCellsPerBatch(self) -> int | None:
        """Max number of cells a single batch covers if the medium is checked"""
        return self._maxCells

    @property
    def upper(self) -> tuple[float, float, float]:
        """Upper corner of the box"""
        return self._upper

    @upper.setter
    def upper(self, value: tuple[float, float, float]) -> None:
        self._upper = value

    @property
    def forwardSourceCode(self) -> str:
        preamble = createPreamble(
            LIGHT_SOURCE_EMIT_PARTICLE=self.emitsParticles,
            LIGHT_SOURCE_CHECK_MEDIUM=self.checkMedium,
            LIGHT_SOURCE_MAX_ATTEMPTS=self.maxAttempts,
            LIGHT_SOURCE_ACCEPT_LOST_RAYS=self.acceptLostRays,
        )
        return preamble + loadShader("lightsource/voxel/forward.glsl")

    def rayCount(self, i: int) -> NDArray[np.uint32]:
        """
        Number of rays created per cell of the batch run using the i-th
        configuration, i.e. element j counts the rays of cell `cellOffset + j`.
        Only available if `checkMedium` is set. The returned array is a view
        into the buffer and gets overwritten by the next batch using the same
        configuration.
        """
        if not self.checkMedium:
            raise RuntimeError("Rays are only counted if checkMedium is set")
        assert self._maxCells is not None
        tensor = self._counts[i]
        return np.ctypeslib.as_array(
            (c_uint32 * self._maxCells).from_address(tensor.memory)
        )

    def _finishParams(self, i: int) -> None:
        super()._finishParams(i)
        lower = np.asarray(self.getParam("lower"), dtype=np.float64)
        upper = np.asarray(self.upper, dtype=np.float64)
        shape = np.asarray(self.getParam("gridShape"), dtype=np.float64)
        if np.any(upper <= lower):
            raise ValueError("upper must be larger than lower in all dimensions")
        self.setParam("_cellSize", tuple((upper - lower) / shape))
        self.setParam("_contribFwd", self.budget)
        if self.checkMedium:
            self.setParam("_countAdr", self._counts[i].address)

    def update(self, i: int) -> None:
        super().update(i)
        # advance after the current offset was written to the device, so the
        # first batch uses the offset as given
        if self.autoAdvance is not None:
            offset = self.getParam("cellOffset")
            self.setParam("cellOffset", offset + self.autoAdvance)

    def run(self, i: int) -> list[hp.Command]:
        if self.checkMedium:
            return [hp.clearTensor(self._counts[i]), *super().run(i)]
        return super().run(i)


def frankTamm(
    wavelength: ArrayLike,
    refractiveIndex: ArrayLike,
    *,
    beta: float = 1.0,
    usePhotonCount: bool = True,
) -> NDArray[np.float64]:
    """Frank Tamm equation
    
    Evaluates the Frank-Tamm equation in units of [m^-1 nm^-1] for the given
    wavelengths and refractive indices:

       d^2 N          alpha  /            1       \\
     --------- = 2pi ------- | 1 - -------------- |
      dx dlam         lam^2 \\      (beta * n)^2  /
    
    if `usePhotonCount` is `True`. Otherwise it samples the emitted energy in
    units of [eV m^-1 nm^-1]:

       d^2 E                            1    /            1       \\
     --------- = pi * c^2 * e * mu_0 ------- | 1 - -------------- |
      dx dlam                         lam^3 \\      (beta * n)^2  /

    where alpha is the finestructure constant and beta = v / c the particle's
    speed relative to the speed of light.
    """
    lam = np.asarray(wavelength) / u.nm
    n = beta * np.asarray(refractiveIndex)
    if usePhotonCount:
        # The 10^9 converts one of the [nm^-1] to [m^-1]
        return 2.0 * np.pi * consts.alpha / lam**2 * (1.0 - (1.0 / n**2)) * 1e9
    else:
        c = np.pi * consts.c**2 * consts.e * consts.mu_0
        return c / lam**3 * (1.0 - (1.0 / n**2)) * 1e18


class CherenkovLightSource(LightSource):
    """
    Light source sampling Cherenkov light from a straight particle track.
    Assumes particle travels at the speed of light (beta = 1.0).

    Parameters
    ----------
    wavelengthSource: WavelengthSource
        Source to sample wavelengths from. Required for forward mode.
    mediumIdx: int
        Index of the medium the light source is located in
    trackStart: (float, float, float), default=(0.0,0.0,0.0)m
        Start position of track
    trackEnd: (float, float, float), default=(100.0,0.0,0.0)m
        End position of track
    startTime: float, default=0.0ns
        Start time of track
    endTime: float, default=100.0 m/c
        End time of track
    usePhotonCount: bool, default=True
        If `True` sampled radiance has units of number photons, otherwise eV
    emitParticles: bool, default=False
        Whether to emit particles instead of rays. If so, `budget` will be
        ignored.
    """

    name = "Cherenkov Light Source"

    class LightParams(Structure):
        _fields_ = [
            ("trackStart", vec3),
            ("startTime", c_float),
            ("trackEnd", vec3),
            ("endTime", c_float),
            ("_trackDir", vec3),
            ("_trackDist", c_float),
            ("mediumIdx", c_uint32),
        ]

    def __init__(
        self,
        wavelengthSource: WavelengthSource | None = None,
        *,
        mediumIdx: int,
        trackStart: tuple[float, float, float] = (0.0, 0.0, 0.0) * u.m,
        trackEnd: tuple[float, float, float] = (100.0, 0.0, 0.0) * u.m,
        startTime: float = 0.0 * u.ns,
        endTime: float = 100.0 * u.m / u.c,
        usePhotonCount: bool = True,
        applyFrankTamm: bool = True,
        emitParticles: bool = False,
    ) -> None:
        # check params
        if applyFrankTamm and emitParticles:
            raise ValueError("Cannot apply Frank-Tamm formula when emitting particles.")
        if emitParticles and not usePhotonCount:
            raise ValueError("Must use photon count when emitting particles")

        lam = wavelengthSource
        draws = LightSourceRNGDraws(
            sampleLightForward=0 if lam is None else lam.nRNGSamples + 2,
            sampleLightBackward=0,
        )
        super().__init__(
            wavelengthSource,
            rngDraws=draws,
            # TODO: implement backward MIS
            params={"LightParams": CherenkovLightSource.LightParams},
        )
        # save params
        self._applyFrankTamm = applyFrankTamm
        self._emitParticles = emitParticles
        self._usePhotonCount = usePhotonCount
        self.setParams(
            trackStart=trackStart,
            trackEnd=trackEnd,
            startTime=startTime,
            endTime=endTime,
            mediumIdx=mediumIdx,
        )

    @property
    def emitsParticles(self) -> bool:
        """Whether this light source emits particles instead of light rays"""
        return self._emitParticles

    @property
    def isFrankTammApplied(self) -> bool:
        """Whether the Frank Tamm equation is applied"""
        return self._applyFrankTamm

    @property
    def usePhotonCount(self) -> bool:
        """Whether to sample radiance in eV or #photons"""
        return self._usePhotonCount

    @property
    def forwardSourceCode(self) -> str | None:
        if self.wavelengthSource is None:
            return None
        preamble = createPreamble(
            LIGHT_SOURCE_EMIT_PARTICLE=self.emitsParticles,
            FRANK_TAMM_USE_ENERGY=not self.usePhotonCount,
            FRANK_TAMM_IS=not self.isFrankTammApplied,
        )
        code = loadShader("lightsource/cherenkov/track/forward.glsl")
        return preamble + code

    @property
    def backwardSourceCode(self) -> str | None:
        if self.emitsParticles:
            return None
        preamble = createPreamble(
            LIGHT_SOURCE_EMIT_PARTICLE=self.emitsParticles,
            FRANK_TAMM_USE_ENERGY=not self.usePhotonCount,
            FRANK_TAMM_IS=not self.isFrankTammApplied,
        )
        code = loadShader("lightsource/cherenkov/track/backward.glsl")
        return preamble + code

    def _finishParams(self, i: int) -> None:
        super()._finishParams(i)
        start = np.array(self.trackStart)
        end = np.array(self.trackEnd)
        dir = end - start
        dist = np.sqrt(np.square(dir).sum(-1))
        dir /= dist
        self.setParams(_trackDir=dir, _trackDist=dist)


class MuonTrackLightSource(LightSource):
    """
    Light source describing the Cherenkov light produced from a high energy
    muon track and its secondaries particles up to 500 MeV in ice or water as
    described in [1]_.

    Parameters
    ----------
    wavelengthSource: WavelengthSource
        Source to sample wavelengths from. Required for forward mode.
    startPosition: (float, float, float), default=(0.0, 0.0, 0.0)
        Start position of the track
    startTime: float, default=0.0ns
        Time at which the muon is at `startPosition`
    endPosition: (float, float, float), default=(0.0, 0.0, 0.0)
        End position of the track
    endTime: float, default=0.0ns
        Time at which the muon is at `endPosition`
    muonEnergy: float, default=1GeV
        Energy of the muon producing secondary particles
    applyFrankTamm: bool, default=True
        Whether to apply the Frank-Tamm equation describing the light yield as a
        function of wavelength.
    emitParticles: bool, default=False
        Whether to emit particles instead of rays. If so, `budget` will be
        ignored.

    .. [1] L. Raedel "Simulation Studies of the Cherenkov Light Yield from
           Relativistic Particles in High-Energy Neutrino Telescopes with
           Geant 4" (2012)
    """

    name = "Muon Track Light Source"

    class TrackParameters(Structure):
        _fields_ = [
            ("startPosition", vec3),
            ("startTime", c_float),
            ("endPosition", vec3),
            ("endTime", c_float),
            ("_direction", vec3),
            ("_dist", c_float),
            ("_energyScale", c_float),
            ("_a_angular", c_float),
            ("_b_angular", c_float),
            ("mediumIdx", c_uint32),
        ]

    def __init__(
        self,
        wavelengthSource: WavelengthSource | None = None,
        *,
        mediumIdx: int,
        startPosition: tuple[float, float, float] = (0.0, 0.0, 0.0),
        startTime: float = 0.0,
        endPosition: tuple[float, float, float] = (0.0, 0.0, 0.0),
        endTime: float = 0.0,
        muonEnergy: float = 1.0 * u.GeV,
        applyFrankTamm: bool = True,
        usePhotonCount: bool = True,
        emitParticles: bool = False,
    ) -> None:
        # check params
        if applyFrankTamm and emitParticles:
            raise ValueError("Cannot apply Frank-Tamm formula when emitting particles.")
        if emitParticles and not usePhotonCount:
            raise ValueError("Must use photon count when emitting particles")

        lam = wavelengthSource
        draws = LightSourceRNGDraws(
            sampleLightForward=0 if lam is None else lam.nRNGSamples + 2,
            sampleLightBackward=1,
        )
        super().__init__(
            wavelengthSource,
            rngDraws=draws,
            # TODO: implement backward MIS
            params={"MuonTrackParams": MuonTrackLightSource.TrackParameters},
            extra={"muonEnergy"},
        )
        # set params
        self.setParams(
            startPosition=startPosition,
            startTime=startTime,
            endPosition=endPosition,
            endTime=endTime,
            mediumIdx=mediumIdx,
        )
        self.muonEnergy = muonEnergy
        self._applyFrankTamm = applyFrankTamm
        self._usePhotonCount = usePhotonCount
        self._emitParticles = emitParticles

    @property
    def emitsParticles(self) -> bool:
        """Whether this light source emits particles instead of light rays"""
        return self._emitParticles

    @property
    def isFrankTammApplied(self) -> bool:
        """Whether the Frank Tamm equation is applied"""
        return self._applyFrankTamm

    @property
    def usePhotonCount(self) -> bool:
        """Whether to sample radiance in eV or #photons"""
        return self._usePhotonCount

    @property
    def muonEnergy(self) -> float:
        """Energy of the muon producing secondary particles"""
        return self._muonEnergy

    @muonEnergy.setter
    def muonEnergy(self, value: float) -> None:
        self._muonEnergy = value
        # calculate energy scale
        self.setParam("_energyScale", 1.1880 + 0.0206 * np.log(value))
        # calculate angular emission profile
        # see notebooks/track_angular_dist_fit.ipynb
        self.setParam("_a_angular", 0.86634 - 7.5624e-3 * np.log10(value))
        self.setParam("_b_angular", 2.5030 + 3.0533e-2 * np.log10(value))

    @property
    def forwardSourceCode(self) -> str | None:
        if self.wavelengthSource is None:
            return None
        preamble = createPreamble(
            FRANK_TAMM_IS=not self.isFrankTammApplied,
            FRANK_TAMM_USE_ENERGY=not self.usePhotonCount,
            LIGHT_SOURCE_EMIT_PARTICLE=self.emitsParticles,
        )
        code = loadShader("lightsource/cherenkov/muon/forward.glsl")
        return preamble + code

    @property
    def backwardSourceCode(self) -> str | None:
        if self.emitsParticles:
            return None
        preamble = createPreamble(
            FRANK_TAMM_IS=not self.isFrankTammApplied,
            FRANK_TAMM_USE_ENERGY=not self.usePhotonCount,
            LIGHT_SOURCE_EMIT_PARTICLE=self.emitsParticles,
        )
        code = loadShader("lightsource/cherenkov/muon/backward.glsl")
        return preamble + code

    def _finishParams(self, i: int) -> None:
        super()._finishParams(i)
        start = np.array(self.startPosition)
        end = np.array(self.endPosition)
        dir = end - start
        dist = np.sqrt(np.square(dir).sum(-1))
        dir /= dist
        self.setParams(_direction=dir, _dist=dist)


class ParticleCascadeLightSource(LightSource):
    """
    Light source describing the Cherenkov light emitted by secondary particles
    in electro-magnetic or hadronic showers in water or ice caused by a primary
    particle above 500 MeV as described in [1]_.

    Parameters
    ----------
    wavelengthSource: WavelengthSource
        Source to sample wavelengths from. Required for forward mode.
    startPosition: (float, float, float), default=(0.0, 0.0, 0.0)
        Start position of the cascade
    startTime: float, default=0.0 ns
        Time point at which the cascade started
    direction: (float, float, float), default=(0.0, 0.0, 1.0)
        Direction in which the cascade evolves, i.e. away from the start point.
    effectiveLength: float, default=0.0
        Length of a Cherenkov track producing the same amount of photons. See
        e.g. eq. 4.7 in [1]_
    a_angular: float, default=0.0
        The a parameter of the angular light emission distribution (eq. 4.5 in
        [1]_).
    b_angular: float, default=0.0
        The b parameter of the angular light emission distribution (eq. 4.5 in
        [1]_)
    a_long: float, default=0.0
        The a parameter of the longitudinal light emission distribution (eq.
        4.10 in [1]_).
    b_long: float, default=0.0
        The b parameter of the longitudinal light emission distribution. Note
        that this differs from eq. 4.10 in [1]_ in order to match the definition
        used by ice tray. Here (and in ice tray), this is the radiation length
        X_0 divided by the b parameter of the underlying gamma distribution.
    applyFrankTamm: bool, default=True
        Whether to apply the Frank-Tamm equation describing the light yield as a
        function of wavelength.
    emitParticles: bool, default=False
        Whether to emit particles instead of rays. If so, `budget` will be
        ignored.

    Note
    ----
    Raedel calculates the radiation length using the following formula:

             1      716.4 [g cm^-2] A
     X_0 = ----- ------------------------
            rho   Z(Z+1) ln(287/sqrt(Z))

    which turns out to be 39.75 cm in ice and 36.08 cm in water [2]_.

    .. [1] L. Raedel "Simulation Studies of the Cherenkov Light Yield from
           Relativistic Particles in High-Energy Neutrino Telescopes with
           Geant 4" (2012)
       [2] L. Raedel, C. Wiebusch: "Calculation of the Cherenkov light yield
           from electromagnetic cascades in ice with Geant4" (2013)
           arXiv:1210.5140v2
    """

    name = "Particle Cascade Light Source"

    class CascadeParameters(Structure):
        _fields_ = [
            ("startPosition", vec3),
            ("startTime", c_float),
            ("direction", vec3),
            ("effectiveLength", c_float),
            ("a_angular", c_float),
            ("b_angular", c_float),
            ("a_long", c_float),
            ("b_long", c_float),
            ("mediumIdx", c_uint32),
        ]

    def __init__(
        self,
        wavelengthSource: WavelengthSource | None = None,
        *,
        mediumIdx: int,
        startPosition: tuple[float, float, float] = (0.0, 0.0, 0.0),
        startTime: float = 0.0 * u.ns,
        direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
        effectiveLength: float = 1.0,
        a_angular: float = 0.0,
        b_angular: float = 0.0,
        a_long: float = 0.0,
        b_long: float = 0.0,
        applyFrankTamm: bool = True,
        usePhotonCount: bool = True,
        emitParticles: bool = False,
    ) -> None:
        # check params
        if applyFrankTamm and emitParticles:
            raise ValueError("Cannot apply Frank-Tamm formula when emitting particles.")
        if emitParticles and not usePhotonCount:
            raise ValueError("Must use photon count when emitting particles")

        lam = wavelengthSource
        draws = LightSourceRNGDraws(
            sampleLightForward=0 if lam is None else lam.nRNGSamples + 34,
            sampleLightBackward=33,
        )
        super().__init__(
            wavelengthSource,
            rngDraws=draws,
            # TODO: implement backward MIS
            params={"CascadeParams": ParticleCascadeLightSource.CascadeParameters},
        )
        # save params
        self.setParams(
            startPosition=startPosition,
            startTime=startTime,
            direction=direction,
            effectiveLength=effectiveLength,
            a_angular=a_angular,
            b_angular=b_angular,
            a_long=a_long,
            b_long=b_long,
            mediumIdx=mediumIdx,
        )
        self._applyFrankTamm = applyFrankTamm
        self._usePhotonCount = usePhotonCount
        self._emitParticles = emitParticles

    @property
    def emitsParticles(self) -> bool:
        """Whether this light source emits particles instead of light rays"""
        return self._emitParticles

    @property
    def isFrankTammApplied(self) -> bool:
        """Whether the Frank Tamm equation is applied"""
        return self._applyFrankTamm

    @property
    def usePhotonCount(self) -> bool:
        """Whether to sample radiance in eV or #photons"""
        return self._usePhotonCount

    @property
    def forwardSourceCode(self) -> str | None:
        if self.wavelengthSource is None:
            return None
        preamble = createPreamble(
            FRANK_TAMM_IS=not self.isFrankTammApplied,
            FRANK_TAMM_USE_ENERGY=not self.usePhotonCount,
            LIGHT_SOURCE_EMIT_PARTICLE=self.emitsParticles,
        )
        code = loadShader("lightsource/cherenkov/cascade/forward.glsl")
        return preamble + code

    @property
    def backwardSourceCode(self) -> str | None:
        if self.emitsParticles:
            return None
        preamble = createPreamble(
            FRANK_TAMM_IS=not self.isFrankTammApplied,
            FRANK_TAMM_USE_ENERGY=not self.usePhotonCount,
            LIGHT_SOURCE_EMIT_PARTICLE=self.emitsParticles,
        )
        code = loadShader("lightsource/cherenkov/cascade/backward.glsl")
        return preamble + code
