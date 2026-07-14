from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from theia.compiler import createPreamble, loadShader
from theia.volume import VolumeModel

from importlib.resources.abc import Traversable
from typing import ClassVar, Type
import json
import re


__all__ = [
    "AbsorbingSurface",
    "BorderSurface",
    "DielectricRoughSurface",
    "DielectricSurface",
    "LambertianReflectingSurface",
    "MetalRoughSurface",
    "MetalSurface",
    "SurfaceModel",
    "SurfaceRNGDraws",
    "ThinSlabSurface",
    "ThinSlabWall",
]


def __dir__():
    return __all__


@dataclass
class SurfaceRNGDraws:
    """Amount of random numbers drawn for each surface model function"""

    prepareSurface: int
    sampleSurfaceInteraction: int
    processSurfaceTargetHit: int

    # MIS

    surfaceScatterRay: int = 0
    sampleSurfaceScattering: int = 0


class SurfaceModel(ABC):
    """
    Base class for all surface models. Surface models implement the interactions
    of a ray with a surface as it hits a geometry.
    """

    models: ClassVar[dict[str, Type[SurfaceModel]]] = {}
    """Mapping containing all registered surface models"""
    name: ClassVar[str] = ""

    def __init_subclass__(cls, *, name: str) -> None:
        if not name:
            raise ValueError("Surface model name cannot be empty")
        if name in SurfaceModel.models:
            raise RuntimeError(f'"{name}" is already in use by another surface model')
        SurfaceModel.models[name] = cls
        cls.name = name

    def __init__(
        self,
        *,
        rngDraws: SurfaceRNGDraws | None,
        # sbtRecord: bytes | None = None,
        supportsMIS: bool = False,
        requiredMediumProperties: set[str] = set(),
        requiredMaterialProperties: set[str] = set(),
    ) -> None:
        if rngDraws is None:
            rngDraws = SurfaceRNGDraws(
                prepareSurface=0,
                sampleSurfaceInteraction=0,
                processSurfaceTargetHit=0,
            )
        self.rngDraws = rngDraws
        # self.sbtRecord = sbtRecord
        self.supportsMIS = supportsMIS
        self.requiredMediumProperties = frozenset(requiredMediumProperties)
        self.requiredMaterialProperties = frozenset(requiredMaterialProperties)

    @property
    def backwardSourceCode(self) -> str | None:
        """
        Source code implementing surface interaction with backward rays.
        `None` if backward tracing is not supported.
        """
        return None

    @property
    def forwardSourceCode(self) -> str | None:
        """
        Source code implementing surface interaction with forward rays.
        `None` if forward tracing is not supported.
        """
        return None

    def save(self, file) -> None:
        # by default just create an empty file
        file.open("w").close()

    @classmethod
    @abstractmethod
    def load(cls, file: Traversable) -> SurfaceModel:
        """Creates a new surface model using the content of the given file"""
        pass

    @staticmethod
    def from_file(file: Traversable) -> SurfaceModel:
        if not file.is_file():
            raise RuntimeError(f'Cannot load config from "{file.name}": Not a file!')
        if file.name not in SurfaceModel.models:
            raise RuntimeError(f'Unknown surface model "{file.name}"')
        # delegate loading to actual model
        cls = SurfaceModel.models[file.name]
        return cls.load(file)

    @abstractmethod
    def __eq__(self, value: object) -> bool:
        pass

    def __hash__(self) -> int:
        return hash(self.name)


class AbsorbingSurface(SurfaceModel, name="absorbing"):
    """Models perfectly absorbing surfaces and detectors."""

    def __init__(self) -> None:
        super().__init__(rngDraws=None)

    @property
    def backwardSourceCode(self) -> str:
        return loadShader("surface/absorbing/backward.glsl")

    @property
    def forwardSourceCode(self) -> str:
        return loadShader("surface/absorbing/forward.glsl")

    @classmethod
    def load(cls, file: Traversable) -> AbsorbingSurface:
        return AbsorbingSurface()

    def __eq__(self, value: object) -> bool:
        return type(value) == AbsorbingSurface

    def __hash__(self) -> int:
        return hash(self.name)


class BorderSurface(SurfaceModel, name="border"):
    """Models border between two surfaces. Transmits all rays without altering them."""

    def __init__(self) -> None:
        super().__init__(rngDraws=None)

    @property
    def backwardSourceCode(self) -> str:
        return loadShader("surface/border/backward.glsl")

    @property
    def forwardSourceCode(self) -> str:
        return loadShader("surface/border/forward.glsl")

    @classmethod
    def load(cls, file: Traversable) -> BorderSurface:
        return BorderSurface()

    def __eq__(self, value: object) -> bool:
        return type(value) == BorderSurface

    def __hash__(self) -> int:
        return hash(self.name)


class DielectricSurface(SurfaceModel, name="dielectric"):
    """
    Models transmission and reflection of perfectly specular surfaces between
    two dielectric media.
    """

    def __init__(self) -> None:
        draws = SurfaceRNGDraws(
            prepareSurface=1,
            sampleSurfaceInteraction=0,
            processSurfaceTargetHit=0,
        )
        super().__init__(
            rngDraws=draws,
            requiredMediumProperties={"refractive_index"},
        )

    @property
    def backwardSourceCode(self) -> str:
        return loadShader("surface/dielectric/backward.glsl")

    @property
    def forwardSourceCode(self) -> str:
        return loadShader("surface/dielectric/forward.glsl")

    @classmethod
    def load(cls, file: Traversable) -> DielectricSurface:
        return DielectricSurface()

    def __eq__(self, value: object) -> bool:
        return type(value) == DielectricSurface

    def __hash__(self) -> int:
        return hash(self.name)
    

class MetalSurface(SurfaceModel, name="metal"):
    """
    Models the perfectly specular reflection of a dielectric-metal interface.
    Metals never transmit; the non-reflected fraction is absorbed.

    The reflectance is either taken from the optional material property
    `reflectivity` (as a function of wavelength) or, if not provided, computed
    from the conductor Fresnel equations using the complex refractive index of
    the metal medium: real part = medium `refractive_index`, imaginary part =
    medium `extinction_coefficient`.
    """

    def __init__(self) -> None:
        draws = SurfaceRNGDraws(
            prepareSurface=1,
            sampleSurfaceInteraction=0,
            processSurfaceTargetHit=0,
        )
        super().__init__(
            rngDraws=draws,
        )

    @property
    def backwardSourceCode(self) -> str:
        return loadShader("surface/metal/backward.glsl")

    @property
    def forwardSourceCode(self) -> str:
        return loadShader("surface/metal/forward.glsl")

    @classmethod
    def load(cls, file: Traversable) -> MetalSurface:
        return MetalSurface()

    def __eq__(self, value: object) -> bool:
        return type(value) == MetalSurface

    def __hash__(self) -> int:
        return hash(self.name)


class MetalRoughSurface(SurfaceModel, name="metal_rough"):
    """
    Models the reflection of a rough dielectric-metal interface. The user can
    chose from several micro-facet based models. Metals never transmit; the
    non-reflected fraction is absorbed.

    The reflected component can be decomposed into a specular spike, specular 
    lobe, diffuse lobe and backscattering via the optional material properties 
    `prob_backscatter`, `prob_specularspike`, `prob_specularlobe` and 
    `prob_diffuselobe`. If these are not given the surface falls back to a pure 
    specular lobe (specular reflection on micro-facets).
    
    The reflectance is either taken from the optional material property
    `reflectivity` or, if not provided, computed from the conductor Fresnel
    equations at the sampled micro-facet using the complex refractive index of
    the metal medium (real part = medium `refractive_index`, imaginary part =
    medium `extinction_coefficient`).

    Parameters
    ----------
   model: str, default="beckmann"
        The micro-facet sampling model. Supported values:

        ``"beckmann"``
            Micro-facets sampled from the Beckmann-Spizzichino distribution.
        ``"trowbridge_reitz"``
            Micro-facets sampled from the Trowbridge-Reitz (GGX) distribution.
        ``"trowbridge_reitz_shadowed"``
            Micro-facets sampled from the Trowbridge-Reitz (GGX) distribution
            with geometric shadowing/masking term.
        ``"gaussian"``
            Micro-facets sampled from a Gaussian (normal) distribution of the
            facet slope angle, using `roughness_parameter` as the Gaussian width
            sigma (in radians). This mirrors the micro-facet sampling of the 
            Geant4 UNIFIED model.
    """

    def __init__(self, *, model="beckmann") -> None:
        supported_models = ["gaussian", "beckmann", "trowbridge_reitz", "trowbridge_reitz_shadowed"]
        if model in supported_models:
            self._model = model
        else:
            raise AttributeError(f"{model} is not a supported metal rough surface model.")

        # Worst-case RNG draws for prepareSurface. The shared retry loop samples up
        # to 8 micro-facets; per retry it draws the facet-normal sampler + 1 (the
        # reflect/transmit decision) + check_microfacet. check_microfacet is free
        # (front-face test) for all models except "trowbridge_reitz_shadowed", which
        # rejection-samples the masking term and thus draws 1 extra per retry:
        #   3 (lobe + diffuse presample) + 8 * (normal + 1 + check) + 1 (fallback)
        #     beckmann / trowbridge_reitz : 8 * (2 + 1 + 0) + 4 = 28
        #     trowbridge_reitz_shadowed   : 8 * (2 + 1 + 1) + 4 = 36
        #     gaussian                    : 8 * (9 + 1 + 0) + 4 = 84
        # "gaussian" additionally rejection-samples the facet angle (up to 4 attempts
        # x 2 draws) inside its normal sampler.
        draws = SurfaceRNGDraws(
            prepareSurface=84 if model == "gaussian" else 36 if model == "trowbridge_reitz_shadowed" else 28,
            sampleSurfaceInteraction=0,
            processSurfaceTargetHit=0,
        )

        # reflectivity and the prob_* lobe weights are optional and thus not
        # listed as required material properties. All models parametrise the
        # roughness through `roughness_parameter`.
        requiredMaterialProperties = {"roughness_parameter"}
        super().__init__(
            rngDraws=draws,
            requiredMediumProperties={"refractive_index"},
            requiredMaterialProperties=requiredMaterialProperties,
        )

    @property
    def model(self) -> str:
        return self._model

    @property
    def backwardSourceCode(self) -> str:
        # the model-specific micro-facet code is prepended to the shared
        # surface implementation and compiled together
        return "\n".join([
            loadShader(f"surface/microfacet_models/{self.model}.glsl"),
            loadShader("surface/metal_rough/backward.glsl"),
        ])

    @property
    def forwardSourceCode(self) -> str:
        return "\n".join([
            loadShader(f"surface/microfacet_models/{self.model}.glsl"),
            loadShader("surface/metal_rough/forward.glsl"),
        ])

    @classmethod
    def load(cls, file: Traversable) -> MetalRoughSurface:
        with file.open() as f:
            config = json.load(f)
        return MetalRoughSurface(model=config["model"])

    def save(self, file) -> None:
        config = {"model": self.model}
        with file.open("w") as f:
            json.dump(config, f)

    def __eq__(self, value: object) -> bool:
        return type(value) == MetalRoughSurface and value.model == self.model

    def __hash__(self) -> int:
        return hash(self.name) ^ hash(self.model)


class DielectricRoughSurface(SurfaceModel, name="dielectric_rough"):
    """
    Models transmission and reflection of rough surfaces between two dielectric
    media. The user can chose from several micro-facet based models.

    By default, the reflectance is computed from the sampled micro-facet. It is
    also possible to specify a wavelength-dependant reflectance via the material
    property `reflectivity`.

    For all models the reflected component can be decomposed into a specular
    spike, specular lobe, diffuse lobe and backscattering via the optional material 
    properties `prob_backscatter`, `prob_specularspike`, `prob_specularlobe` and 
    `prob_diffuselobe`. If these are not given the surface falls back to a pure 
    specular lobe (specular reflection on micro-facets).

    The transmitted component always gets refracted according to Snell's law at
    the micro-facet.

    Parameters
    ----------
    model: str, default="beckmann"
        The micro-facet sampling model. Supported values:

        ``"beckmann"``
            Micro-facets sampled from the Beckmann-Spizzichino distribution.
        ``"trowbridge_reitz"``
            Micro-facets sampled from the Trowbridge-Reitz (GGX) distribution.
        ``"trowbridge_reitz_shadowed"``
            Micro-facets sampled from the Trowbridge-Reitz (GGX) distribution
            with geometric shadowing/masking term.
        ``"gaussian"``
            Micro-facets sampled from a Gaussian (normal) distribution of the
            facet slope angle, using `roughness_parameter` as the Gaussian width
            sigma (in radians). This mirrors the micro-facet sampling of the 
            Geant4 UNIFIED model.

    Note
    ----
    The Geant4 UNIFIED SpecularLobe is essentially the small-angle approximation of
    the Beckmann model with `sigma_alpha = roughness_parameter / sqrt(2)`. For small
    roughnesses, these models will give approximately the same result.
    """

    def __init__(self, *, model = "beckmann") -> None:
        supported_models = ["gaussian", "beckmann", "trowbridge_reitz", "trowbridge_reitz_shadowed"]
        if model in supported_models:
            self._model = model
        else:
            raise AttributeError(f"{model} is not a supported dielectric rough surface model.")

        # Worst-case RNG draws for prepareSurface. The shared retry loop samples up
        # to 8 micro-facets; per retry it draws the facet-normal sampler + 1 (the
        # reflect/transmit decision) + check_microfacet. check_microfacet is free
        # (front-face test) for all models except "trowbridge_reitz_shadowed", which
        # rejection-samples the masking term and thus draws 1 extra per retry:
        #   3 (lobe + diffuse presample) + 8 * (normal + 1 + check) + 1 (fallback)
        #     beckmann / trowbridge_reitz : 8 * (2 + 1 + 0) + 4 = 28
        #     trowbridge_reitz_shadowed   : 8 * (2 + 1 + 1) + 4 = 36
        #     gaussian                    : 8 * (9 + 1 + 0) + 4 = 84
        # "gaussian" additionally rejection-samples the facet angle (up to 4 attempts
        # x 2 draws) inside its normal sampler.
        draws = SurfaceRNGDraws(
            prepareSurface=84 if model == "gaussian" else 36 if model == "trowbridge_reitz_shadowed" else 28,
            sampleSurfaceInteraction=0,
            processSurfaceTargetHit=0,
        )

        # reflectivity and the prob_* lobe weights are optional and thus not
        # listed as required material properties. All models parametrise the
        # roughness through `roughness_parameter`.
        requiredMaterialProperties = {"roughness_parameter"}
        super().__init__(
            rngDraws=draws,
            requiredMediumProperties={"refractive_index"},
            requiredMaterialProperties=requiredMaterialProperties,
        )

    @property
    def model(self) -> str:
        return self._model

    @property
    def backwardSourceCode(self) -> str:
        # the model-specific micro-facet code is prepended to the shared
        # surface implementation and compiled together
        return "\n".join([
            loadShader(f"surface/microfacet_models/{self.model}.glsl"),
            loadShader("surface/dielectric_rough/backward.glsl"),
        ])

    @property
    def forwardSourceCode(self) -> str:
        return "\n".join([
            loadShader(f"surface/microfacet_models/{self.model}.glsl"),
            loadShader("surface/dielectric_rough/forward.glsl"),
        ])

    @classmethod
    def load(cls, file: Traversable) -> DielectricRoughSurface:
        with file.open() as f:
            config = json.load(f)
        model = config["model"]
        # backwards compatibility: "unified" was renamed to "gaussian"
        if model == "unified":
            model = "gaussian"
        return DielectricRoughSurface(model=model)
    
    def save(self, file) -> None:
        config = {"model": self.model}
        with file.open("w") as f:
            json.dump(config, f)

    def __eq__(self, value: object) -> bool:
        return type(value) == DielectricRoughSurface and value.model == self.model

    def __hash__(self) -> int:
        return hash(self.name) ^ hash(self.model)


class LambertianReflectingSurface(SurfaceModel, name="lambert"):
    """
    Models perfectly diffuse reflecting surfaces.
    Reflectance defaults to 100%, but can be specified as function of wavelength
    in the `REFLECTIVITY` property.
    """

    def __init__(self) -> None:
        draws = SurfaceRNGDraws(
            prepareSurface=1,
            sampleSurfaceInteraction=2,
            processSurfaceTargetHit=0,
            sampleSurfaceScattering=2,
        )
        super().__init__(
            rngDraws=draws,
            requiredMaterialProperties={"reflectivity"},
            supportsMIS=True,
        )

    @property
    def backwardSourceCode(self) -> str | None:
        return loadShader("surface/lambert/backward.glsl")

    @property
    def forwardSourceCode(self) -> str | None:
        return loadShader("surface/lambert/forward.glsl")

    @classmethod
    def load(cls, file: Traversable) -> LambertianReflectingSurface:
        return LambertianReflectingSurface()

    def __eq__(self, value: object) -> bool:
        return type(value) == LambertianReflectingSurface

    def __hash__(self) -> int:
        return hash(self.name)


# supported micro-facet models (shared with the rough surfaces above)
_MICROFACET_MODELS = [
    "gaussian",
    "beckmann",
    "trowbridge_reitz",
    "trowbridge_reitz_shadowed",
]
# global symbols defined by the micro-facet snippets that must be namespaced
# when a wall model is compiled twice into the same shader
_MICROFACET_SYMBOLS = [
    "sample_microfacet_normal",
    "check_microfacet",
    "masking_function",  # only defined by trowbridge_reitz_shadowed
]
# global symbols defined by a surface-model template; namespaced per wall so the
# two walls (and the thin-slab's own SurfaceProperties) do not clash
_SURFACE_SYMBOLS = [
    "SurfaceProperties",
    "prepareSurface",
    "sampleSurfaceInteraction",
    "processSurfaceTargetHit",
    "SURFACE_MODEL_SPECULAR",
]
# global symbols defined by the volume models; namespaced with a `slab_` prefix
# so they do not clash with the tracer's own volume proxy in the same shader
_VOLUME_SYMBOLS = [
    "VolumeProperties",
    "prepareVolume",
    "sampleFluorescenceWavelengthShift",
    "sampleInteractionLength",
    "applyVolumeSampled",
    "applyVolume",
    "sampleVolumeInteraction",
    "volumeScatterRay",
    "sampleVolumeScattering",
    "volumeScatterProb",
]
# preamble macros a volume model may set; undefined again after embedding so they
# do not leak into the tracer's volume proxy (included later in the same shader)
_VOLUME_MACROS = [
    "VOLUME_ABSORB_RAY",
    "VOLUME_MODEL_NO_SCATTERING",
    "FLUORESCENCE_NO_WAVELENGTH_UPSHIFT",
    "FLUORESCENCE_EXPONENTIAL_DECAY",
]


def _renameGuard(src: str, prefix: str) -> str:
    """Uniquify the include guard of a shader snippet so it can be embedded next
    to another copy of the same file in one translation unit."""
    m = re.search(r"#ifndef\s+(_INCLUDE\w*)", src)
    if m is not None:
        guard = m.group(1)
        # strip leading underscores to avoid reserved "__" identifiers
        src = src.replace(guard, prefix + guard.lstrip("_"))
    return src


@dataclass(frozen=True)
class ThinSlabWall:
    """
    Description of a single wall of a `ThinSlabSurface`.

    Parameters
    ----------
    type: str, default="dielectric"
        Physical model of the wall. One of:

        ``"dielectric"``
            Fresnel reflection/transmission between the interior and exterior
            medium (the only type that lets rays leave/enter the slab).
        ``"metal"``
            Conductor reflection (never transmits, the non-reflected fraction is
            absorbed). The metal is the medium on the far side of the wall; its
            `refractive_index` and `extinction_coefficient` form the complex index
            used in the conductor Fresnel equations. Models e.g. a
            dielectric-layer/metal interface (the ray reflects off the metal but
            never enters it).
        ``"absorbing"``
            Perfectly absorbing/black wall.
    model: str | None, default="beckmann"
        Micro-facet model used to perturb the wall normal (see
        `DielectricRoughSurface`). ``None`` gives a perfectly smooth wall.
        Ignored for ``type="absorbing"``.
    """

    type: str = "dielectric"
    model: str | None = "beckmann"

    def __post_init__(self) -> None:
        if self.type not in ("dielectric", "metal", "absorbing"):
            raise AttributeError(f"{self.type} is not a supported thin-slab wall type.")
        if self.model is not None and self.model not in _MICROFACET_MODELS:
            raise AttributeError(f"{self.model} is not a supported micro-facet model.")

    @property
    def isRough(self) -> bool:
        return self.type != "absorbing" and self.model is not None


class ThinSlabSurface(SurfaceModel, name="thin_slab"):
    """
    Surface model representing a thin, plane-parallel layer (two walls and a
    volume in between) as a *single* scene surface.

    Very thin volumes cannot be simulated as regular geometry in Theia: the
    single-precision ray offset applied after every surface interaction is
    coarser than the layer and makes the ray overshoot the opposite wall. This
    model instead simulates the whole multiple-scattering transport inside the
    layer analytically in a local 1D coordinate (the height along the surface
    normal, tangential motion is neglected) during a single surface interaction,
    and only returns the ray once it leaves the slab. No ray offsetting or
    re-intersection of the thin geometry is required.

    The two walls can use independent models (`outside`/`inside`); the interior
    volume is described by any forward-capable `VolumeModel`.

    Only forward tracing is supported. NEE/MIS is not supported.

    Parameters
    ----------
    volume: VolumeModel | str
        Volume model of the interior. Passing a string default-constructs the
        registered model of that name.
    outside: ThinSlabWall, default=ThinSlabWall()
        Model of the wall on the geometry's outside (the material ``outside``
        side). Which wall the ray hits first depends on its approach direction.
    inside: ThinSlabWall | None, default=None
        Model of the wall on the geometry's inside. If ``None`` the `outside`
        model is used for both walls.
    maxIterations: int, default=128
        Maximum number of internal segments (free flights + wall interactions)
        simulated before the ray is terminated. Bounds run time and RNG usage.

    Required material properties
    ----------------------------
    - ``slab_thickness`` (`FloatProperty`): layer thickness in world units.
    - ``slab_medium`` (`IntProperty`): media-table index of the interior medium,
      stored in the first component, e.g. ``IntProperty((store.media[name], 0))``.
    - ``roughness_outside`` / ``roughness_inside`` (`FloatProperty`) for rough
      walls (the Gaussian width sigma for the ``"gaussian"`` model).
    For ``"metal"`` walls the medium on the far side of that wall must be defined
    with both ``refractive_index`` and ``extinction_coefficient`` (the complex
    index of the metal).

    Note
    ----
    The interior medium must be registered in the `MaterialStore` (e.g. passed
    via its ``media=`` argument) so its property tables are uploaded; the index
    stored in ``slab_medium`` must match ``store.media[<name>]``. The interior
    medium's own ``physicModel`` is irrelevant (this model performs the
    transport); ``Transparent`` is a sensible choice.
    """

    def __init__(
        self,
        *,
        volume: VolumeModel | str,
        outside: ThinSlabWall = ThinSlabWall(),
        inside: ThinSlabWall | None = None,
        maxIterations: int = 128,
    ) -> None:
        if isinstance(volume, str):
            if volume not in VolumeModel.models:
                raise AttributeError(f"{volume} is not a known volume model.")
            volume = VolumeModel.models[volume]()
        if volume.forwardSourceCode is None:
            raise AttributeError("The volume model does not support forward tracing.")
        if inside is None:
            inside = outside
        if maxIterations < 1:
            raise AttributeError("maxIterations must be a positive integer.")

        self._volume = volume
        self._outside = outside
        self._inside = inside
        self._maxIter = int(maxIterations)

        # assemble required properties
        reqMat = {"slab_thickness", "slab_medium"}
        reqMed = {"refractive_index", *volume.requiredMediumProperties}
        for wall, suffix in ((outside, "outside"), (inside, "inside")):
            if wall.type == "metal":
                # the metal is the medium on the far side of the wall; its complex
                # index (refractive_index + extinction_coefficient) is used
                reqMed |= {"refractive_index", "extinction_coefficient"}
            if wall.isRough:
                reqMat.add(f"roughness_{suffix}")

        draws = SurfaceRNGDraws(
            prepareSurface=0,
            # generous upper bound: each internal segment may draw a handful of
            # random numbers (free flight, scatter, wall decision, micro-facet)
            sampleSurfaceInteraction=self._maxIter * 8,
            processSurfaceTargetHit=0,
        )
        super().__init__(
            rngDraws=draws,
            supportsMIS=False,
            requiredMediumProperties=reqMed,
            requiredMaterialProperties=reqMat,
        )

    @property
    def volume(self) -> VolumeModel:
        return self._volume

    @property
    def outside(self) -> ThinSlabWall:
        return self._outside

    @property
    def inside(self) -> ThinSlabWall:
        return self._inside

    @property
    def maxIterations(self) -> int:
        return self._maxIter

    def _volumeSource(self) -> str:
        """Embed the volume model with all its public symbols namespaced
        (`slab_` prefix) and its preamble macros scoped to this include."""
        src = self._volume.forwardSourceCode
        src = _renameGuard(src, "SLAB_")
        defs = "\n".join(f"#define {s} slab_{s}" for s in _VOLUME_SYMBOLS)
        undefs = "\n".join(f"#undef {s}" for s in _VOLUME_SYMBOLS)
        # scope volume preamble macros so they do not leak into the volume proxy
        macro_undefs = "\n".join(
            f"#ifdef {m}\n#undef {m}\n#endif" for m in _VOLUME_MACROS
        )
        return f"{defs}\n{src}\n{undefs}\n{macro_undefs}\n"

    def _wallModelSource(self, wall: ThinSlabWall, suffix: str) -> str:
        """Embed the reused surface-model source for a single wall with all its
        public symbols namespaced per wall and (for rough walls) the roughness
        lookup redirected to the per-wall material slot."""
        rough = wall.isRough
        if wall.type == "absorbing":
            template = "surface/absorbing/forward.glsl"
        elif wall.type == "metal":
            template = "surface/metal_rough/template.glsl" if rough else "surface/metal/template.glsl"
        else:
            template = "surface/dielectric_rough/template.glsl" if rough else "surface/dielectric/template.glsl"

        srcParts = []
        if rough:
            srcParts.append(loadShader(f"surface/microfacet_models/{wall.model}.glsl"))
        srcParts.append(loadShader(template))
        src = "\n".join(srcParts)

        # namespace all symbols the model defines (functions, struct, macros) so
        # top and bottom (and the slab's own SurfaceProperties) never collide
        symbols = list(_SURFACE_SYMBOLS)
        if rough:
            symbols += _MICROFACET_SYMBOLS
        for s in symbols:
            src = re.sub(rf"\b{re.escape(s)}\b", f"{s}_{suffix}", src)
        # the micro-facet snippet carries an include guard -> uniquify it
        src = _renameGuard(src, f"SLAB_{suffix.upper()}_")

        # redirect the roughness lookup to the per-wall material slot
        if rough:
            slot = "ROUGHNESS_PARAMETER"
            wallSlot = "ROUGHNESS_" + suffix.upper()
            pre = "\n".join([
                f"#ifdef MATERIAL_SLOT_{slot}",
                f"#undef MATERIAL_SLOT_{slot}",
                "#endif",
                f"#define MATERIAL_SLOT_{slot} MATERIAL_SLOT_{wallSlot}",
            ])
            src = f"{pre}\n{src}\n#undef MATERIAL_SLOT_{slot}\n"
        return src + "\n"

    def _wallAdapterSource(self, wall: ThinSlabWall, suffix: str) -> str:
        """Instantiate the adapter `surface/thin_slab/wall.glsl` for a wall."""
        lines = [
            f"#define WALL_FN slabWall_{suffix}",
            f"#define WALL_SAMPLE sampleSurfaceInteraction_{suffix}",
        ]
        undef = ["#undef WALL_FN", "#undef WALL_SAMPLE"]
        # models without a SurfaceProperties/prepareSurface pair (e.g. absorbing)
        # use the props-less path in the adapter (#ifdef WALL_PROPS)
        if wall.type != "absorbing":
            lines += [
                f"#define WALL_PREPARE prepareSurface_{suffix}",
                f"#define WALL_PROPS SurfaceProperties_{suffix}",
            ]
            undef += ["#undef WALL_PREPARE", "#undef WALL_PROPS"]
        lines.append('#include "surface/thin_slab/wall.glsl"')
        lines += undef
        return "\n".join(lines) + "\n"

    @property
    def forwardSourceCode(self) -> str:
        parts = [
            createPreamble(
                THIN_SLAB_MAX_ITER=self._maxIter,
                THIN_SLAB_WALL_REFLECT=0,
                THIN_SLAB_WALL_TRANSMIT=1,
                THIN_SLAB_WALL_ABSORB=2,
            ),
            # reflectRay/transmitRay (+ offset) used by the reused wall templates
            '#include "surface/propagate/forward.glsl"',
            # embed the volume model first: some volume wrappers (#define RAY ... /
            # #undef RAY) would otherwise clobber the RAY macro set below
            self._volumeSource(),
            # macros the reused wall templates expect from their forward wrapper
            "#define RAY ForwardRay",
            "#define NO_REFLECT_BIT MATERIAL_NO_REFLECT_FWD_BIT",
            "#define NO_TRANSMIT_BIT MATERIAL_NO_TRANSMIT_FWD_BIT",
        ]
        for wall, suffix in ((self._outside, "outside"), (self._inside, "inside")):
            parts.append(self._wallModelSource(wall, suffix))
            parts.append(self._wallAdapterSource(wall, suffix))
        # do not leak the template macros into the rest of the tracer
        parts += ["#undef RAY", "#undef NO_REFLECT_BIT", "#undef NO_TRANSMIT_BIT"]
        parts.append(loadShader("surface/thin_slab/forward.glsl"))
        return "\n".join(parts)

    # backward tracing is not supported -> inherit backwardSourceCode == None

    @classmethod
    def load(cls, file: Traversable) -> ThinSlabSurface:
        with file.open() as f:
            config = json.load(f)
        return ThinSlabSurface(
            volume=VolumeModel.models[config["volume"]](),
            outside=ThinSlabWall(**config["outside"]),
            inside=ThinSlabWall(**config["inside"]),
            maxIterations=config["maxIterations"],
        )

    def save(self, file) -> None:
        config = {
            "volume": self._volume.name,
            "outside": {"type": self._outside.type, "model": self._outside.model},
            "inside": {"type": self._inside.type, "model": self._inside.model},
            "maxIterations": self._maxIter,
        }
        with file.open("w") as f:
            json.dump(config, f)

    def __eq__(self, value: object) -> bool:
        return (
            type(value) == ThinSlabSurface
            and value._volume == self._volume
            and value._outside == self._outside
            and value._inside == self._inside
            and value._maxIter == self._maxIter
        )

    def __hash__(self) -> int:
        return hash((self.name, self._volume, self._outside, self._inside, self._maxIter))
