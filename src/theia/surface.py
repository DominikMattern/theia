from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from theia.compiler import loadShader

from importlib.resources.abc import Traversable
from typing import ClassVar, Type
import json


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
            facet slope angle via a pre-computed ppf table (generated by
            `property.RoughUnifiedSurfacePPFProperty`). This mirrors the
            micro-facet sampling of the Geant4 UNIFIED model.
    """

    def __init__(self, *, model="beckmann") -> None:
        draws = SurfaceRNGDraws(
            prepareSurface=36,
            sampleSurfaceInteraction=0,
            processSurfaceTargetHit=0,
        )

        supported_models = ["gaussian", "beckmann", "trowbridge_reitz", "trowbridge_reitz_shadowed"]
        if model in supported_models:
            self._model = model
        else:
            raise AttributeError(f"{model} is not a supported metal rough surface model.")

        # reflectivity and the prob_* lobe weights are optional and thus not
        # listed as required material properties
        if model == "gaussian":
            requiredMaterialProperties = {"ppf"}
        else:
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
            facet slope angle via a pre-computed ppf table (generated by
            `property.RoughUnifiedSurfacePPFProperty`). This mirrors the
            micro-facet sampling of the Geant4 UNIFIED model.

    Note
    ----
    The Geant4 UNIFIED SpecularLobe is essentially the small-angle approximation of
    the Beckmann model with `sigma_alpha = roughness_parameter / sqrt(2)`. For small
    roughnesses, these models will give approximately the same result.
    """

    def __init__(self, *, model = "beckmann") -> None:
        draws = SurfaceRNGDraws(
            prepareSurface=36,
            sampleSurfaceInteraction=0,
            processSurfaceTargetHit=0,
        )

        supported_models = ["gaussian", "beckmann", "trowbridge_reitz", "trowbridge_reitz_shadowed"]
        if model in supported_models:
            self._model = model
        else:
            raise AttributeError(f"{model} is not a supported dielectric rough surface model.")

        # reflectivity and the prob_* lobe weights are optional and thus not
        # listed as required material properties
        if model == "gaussian":
            requiredMaterialProperties = {"ppf"}
        else:
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
