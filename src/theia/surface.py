from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import json

import numpy as np

from theia.compiler import createPreamble, createPreamble, loadShader
from theia.microfacet import (
    MICROFACET_MODELS,
    emptyNormalizationTable,
    fillNormalizationTables,
    gaussianNormalization,
)
from theia.property import FloatProperty, Property, TableProperty
from theia.volume import Fluorescent, VolumeModel

from collections.abc import Mapping
from importlib.resources.abc import Traversable
from numpy.typing import NDArray
from typing import ClassVar, Type
import json
import re
import warnings

__all__ = [
    "AbsorbingSurface",
    "BorderSurface",
    "DielectricRoughSurface",
    "DielectricSurface",
    "LambertianReflectingSurface",
    "MetallicSurface",
    "MetalRoughSurface",
    "MetalSurface",
    "SurfaceModel",
    "SurfaceRNGDraws",
    "ThinDielectricSurface",
    "ThinMetallicSurface",
    "ThinSlabSurface",
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

    def derivedMaterialProperties(
        self, material, media: Mapping[str, object]
    ) -> Mapping[str, Property]:
        """
        Properties the model computes itself from the material's own properties
        and injects into the material table.

        Called by `MaterialStore` for every material referencing this model.
        The returned properties are not stored on the `Material` and therefore
        not serialized: they are reconstructed on the next upload. Returning a
        name the material already defines is an error.

        Parameters
        ----------
        material: Material
            Material referencing this model.
        media: { str, Medium }
            All media of the store by name, so the model can reach the
            properties of the media the material separates.

        Returns
        -------
        Additional properties keyed by slot name. Empty by default.
        """
        return {}

    def finalizeMaterials(self, store, materialIndices: list[int]) -> None:
        """
        Called once by `MaterialStore` after its tables are complete, with the
        indices of all materials referencing this model.

        Unlike `derivedMaterialProperties` this may run device work, so it is
        where a property can be computed that needs the finished tables as its
        input - something the property itself cannot do without a cycle. Storage
        reserved by `derivedMaterialProperties` can be filled in place.

        Does nothing by default.
        """
        pass

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


def _refractiveIndexAt(medium, lam: NDArray) -> NDArray:
    """Refractive index of `medium` at the given wavelengths"""
    if medium is None:
        return np.ones_like(lam)  # vacuum
    # string key rather than OpticalProperties: material.py imports this module
    prop = medium.properties.get("refractive_index")
    if isinstance(prop, TableProperty) and prop.table is not None:
        lo, hi = prop.table.axisRanges[0]
        samples = prop.table.samples
        return np.interp(lam, np.linspace(lo, hi, len(samples)), samples)
    if isinstance(prop, FloatProperty):
        return np.full_like(lam, prop.value[1])
    return np.ones_like(lam)


def _etaRange(material, media: Mapping[str, object]) -> tuple[float, float]:
    """
    Range of the inward ratio `e = n_outside/n_inside` at `material`.

    The two indices have to be taken at the *same* wavelength - a ray never sees
    the outside index at one wavelength and the inside index at another. Pairing
    them keeps the range narrow (often something like 1.4 to 1.5) instead of the
    much wider hull of the two indices taken independently. Both traversal
    directions share this axis: inward sees `eta = e`, outward `eta = 1/e`.
    """
    resolve = lambda m: media.get(m) if isinstance(m, str) else m
    inside, outside = resolve(material.inside), resolve(material.outside)

    ranges = [m.wavelengthRange for m in (inside, outside) if m is not None]
    if not ranges:
        return (0.99, 1.01)  # vacuum on both sides: no interface at all
    lam = np.linspace(max(r[0] for r in ranges), min(r[1] for r in ranges), 128)

    e = _refractiveIndexAt(outside, lam) / _refractiveIndexAt(inside, lam)
    # a little margin so the lookup never has to clamp at the very edge
    return (float(e.min()) * 0.99, float(e.max()) * 1.01)


def _roughDerivedProperties(
    model: str,
    supportNEE: bool,
    material,
    media: Mapping[str, object],
    *,
    transmission: bool,
) -> dict[str, Property]:
    """
    Builds the normalization properties the rough surface models need for NEE.
    Shared by `MetalRoughSurface` and `DielectricRoughSurface`.
    """
    # without NEE the model is marked specular, so nothing reads these slots and
    # there is no reason to spend time building them
    if not supportNEE:
        return {}

    props: dict[str, Property] = {}

    # The Gaussian facet density has no closed form normalization, so it has to
    # be precomputed here. This is the only place left where the host needs to
    # know how a micro-facet model parametrizes itself - everything else the
    # models load themselves in prepare_microfacet().
    if model == "gaussian":
        sigma = material.properties.get("roughness_parameter")
        if not isinstance(sigma, FloatProperty):
            raise ValueError(
                f'Material "{material.name}" must define "roughness_parameter" as '
                f"a FloatProperty to be used with the gaussian micro-facet model"
            )
        props["microfacet_z_sigma"] = FloatProperty(
            gaussianNormalization(float(sigma.value[1]))
        )

    # The tables themselves are computed on the device by `finalizeMaterials`
    # once this table is complete - the acceptance rate depends on the roughness
    # and the refractive indices, which only become readable then. Here we only
    # reserve the storage with the right axes.
    if transmission:
        etaRange = _etaRange(material, media)
        # one table per traversal direction, both indexed by the inward ratio
        props["microfacet_z_inward"] = TableProperty(emptyNormalizationTable(etaRange))
        props["microfacet_z_outward"] = TableProperty(emptyNormalizationTable(etaRange))
        # With the wavelengths paired the range only contains n_i == n_o if the
        # two media genuinely cross. There the interface is optically absent and
        # the acceptance rate spikes, which no reasonable grid resolves.
        if etaRange[0] < 1.0 < etaRange[1]:
            warnings.warn(
                f'Material "{material.name}" has media whose refractive indices '
                f"cross within their shared wavelength range (n_outside/n_inside "
                f"spans {etaRange[0]:.4f}..{etaRange[1]:.4f}). Where they are equal "
                f"the interface is optically absent and the acceptance rate of the "
                f"micro-facet model spikes; NEE contributions near that wavelength "
                f"may be off by a few percent."
            )
    else:
        props["microfacet_z_refl"] = TableProperty(emptyNormalizationTable())

    return props


class MetalRoughSurface(SurfaceModel, name="metal_rough"):
    """
    Models the reflection of a rough dielectric-metal interface. The user can
    chose from several micro-facet based models. Metals never transmit; the
    non-reflected fraction is absorbed.

    Except for `"unified"`, the surface is a pure specular lobe on micro-facets
    and supports NEE/MIS.

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
        ``"unified"``
            The Geant4 UNIFIED model itself: Gaussian micro-facets, the
            reflection split into a specular spike, specular lobe, diffuse lobe
            and backscattering (weighted by the optional material properties
            `prob_backscatter`, `prob_specularspike`, `prob_specularlobe` and
            `prob_diffuselobe`), and Geant4's walk across the micro structure in
            place of a facet rejection. Contains delta components and has no
            tractable density, so it is excluded from NEE/MIS. Pick it only if
            you need the Geant4 behaviour.
    supportNEE: bool, default=True
        Whether to build the tables NEE/MIS needs. Has no effect for
        `model="unified"`, which never supports NEE.
    """

    def __init__(
        self,
        *,
        model="beckmann",
        supportNEE: bool = True,
    ) -> None:
        if model not in MICROFACET_MODELS and model != "unified":
            raise AttributeError(f"{model} is not a supported metal rough surface model.")
        self._model = model
        # the UNIFIED model has delta components a shadow ray cannot reach and a
        # walk without a closed form density, so NEE is unavailable there
        self._supportNEE = bool(supportNEE) and model != "unified"

        # Worst-case RNG draws for prepareSurface. The facet loop runs 21 times,
        # the last one being the fall-back, which draws neither a facet nor an
        # acceptance decision. The normal sampler draws 2, except "gaussian"
        # (and hence "unified") which rejection-samples the facet angle (up to 4
        # attempts x 2 draws) plus 1 for the azimuth.
        sampler = 9 if model in ("gaussian", "unified") else 2
        if model == "unified":
            #   3 (lobe choice + pre-sampled diffuse direction)
            # + 20 * normal    (the walk draws nothing else; the reflectance is a
            #                   weight, and the fall-back draws no normal)
            # + 1 (reflect/absorb coin, drawn once after the walk)
            prepare = 3 + 20 * sampler + 1
        else:
            #   20 * (normal + facet acceptance) + 1 (acceptance in the fall-back)
            # + 1 (reflect/absorb coin, drawn once after the loop)
            prepare = 20 * (sampler + 1) + 2
        draws = SurfaceRNGDraws(
            prepareSurface=prepare,
            sampleSurfaceInteraction=0,
            processSurfaceTargetHit=0,
            # the NEE proposal draws one facet plus a branch decision
            sampleSurfaceScattering=sampler + 1 if self._supportNEE else 0,
        )

        # reflectivity and the prob_* lobe weights are optional and thus not
        # listed as required material properties. All models parametrise the
        # roughness through `roughness_parameter`.
        requiredMaterialProperties = {"roughness_parameter"}
        super().__init__(
            rngDraws=draws,
            supportsMIS=self._supportNEE,
            requiredMediumProperties={"refractive_index"},
            requiredMaterialProperties=requiredMaterialProperties,
        )

    @property
    def model(self) -> str:
        return self._model

    @property
    def supportNEE(self) -> bool:
        """Whether this surface can be reached by next event estimation"""
        return self._supportNEE

    def _sourceCode(self, mode: str) -> str:
        # the micro-facet code is prepended to the shared surface implementation
        # and compiled together
        if self.model == "unified":
            # the UNIFIED model is always gaussian and marks itself specular, so
            # it needs neither a model choice nor the NEE config header
            return "\n".join([
                loadShader("surface/microfacet_models/gaussian.glsl"),
                loadShader(f"surface/unified/{mode}.glsl"),
            ])
        return "\n".join([
            createPreamble(SURFACE_ROUGH_NO_NEE=not self._supportNEE),
            # resolves ROUGH_SURFACE_NEE; has to come before the model, which
            # keys its NEE only parts off it
            loadShader("surface/rough_nee_config.glsl"),
            loadShader(f"surface/microfacet_models/{self.model}.glsl"),
            loadShader(f"surface/metal_rough/{mode}.glsl"),
        ])

    @property
    def backwardSourceCode(self) -> str:
        return self._sourceCode("backward")

    @property
    def forwardSourceCode(self) -> str:
        return self._sourceCode("forward")

    def derivedMaterialProperties(
        self, material, media: Mapping[str, object]
    ) -> Mapping[str, Property]:
        return _roughDerivedProperties(
            self.model,
            self._supportNEE,
            material,
            media,
            transmission=False,
        )

    def finalizeMaterials(self, store, materialIndices: list[int]) -> None:
        if not self._supportNEE:
            return
        fillNormalizationTables(
            store,
            self.forwardSourceCode,
            materialIndices,
            [("microfacet_z_refl", False)],
        )

    @classmethod
    def load(cls, file: Traversable) -> MetalRoughSurface:
        with file.open() as f:
            config = json.load(f)
        return MetalRoughSurface(
            model=config["model"],
            supportNEE=config.get("supportNEE", True),
        )

    def save(self, file) -> None:
        config = {
            "model": self.model,
            "supportNEE": self._supportNEE,
        }
        with file.open("w") as f:
            json.dump(config, f)

    def __eq__(self, value: object) -> bool:
        return (
            type(value) == MetalRoughSurface
            and value.model == self.model
            and value.supportNEE == self.supportNEE
        )

    def __hash__(self) -> int:
        return hash(self.name) ^ hash(self.model) ^ hash(self.supportNEE)


class DielectricRoughSurface(SurfaceModel, name="dielectric_rough"):
    """
    Models transmission and reflection of rough surfaces between two dielectric
    media. The user can chose from several micro-facet based models.

    Except for `"unified"`, the surface is a pure specular lobe on micro-facets
    and supports NEE/MIS. The reflectance is always computed from the Fresnel
    equations at the sampled micro-facet; the transmitted component gets refracted
    according to Snell's law at the micro-facet.

    All material flag combinations show the same reflection behaviour. Forbidding
    a channel only absorbs what the surface may no longer emit, it does not
    change how the surface reflects.

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
        ``"unified"``
            The Geant4 UNIFIED model itself: Gaussian micro-facets, the
            reflection split into a specular spike, specular lobe, diffuse lobe
            and backscattering (weighted by the optional material properties
            `prob_backscatter`, `prob_specularspike`, `prob_specularlobe` and
            `prob_diffuselobe`), and Geant4's walk across the micro structure in
            place of a facet rejection - a ray that fails to leave keeps the
            direction the facet gave it, and may cross the interface any number of
            times before it does. Contains delta components and has no tractable
            density, so it is excluded from NEE/MIS. Pick it only if you need the
            Geant4 behaviour.
    supportNEE: bool, default=True
        Whether to build the tables NEE/MIS needs. Has no effect for
        `model="unified"`, which never supports NEE.

    Note
    ----
    The Geant4 UNIFIED SpecularLobe is essentially the small-angle approximation of
    the Beckmann model with `sigma_alpha = roughness_parameter / sqrt(2)`. For small
    roughnesses, these models will give approximately the same result.
    """

    def __init__(
        self,
        *,
        model="beckmann",
        supportNEE: bool = True,
    ) -> None:
        if model not in MICROFACET_MODELS and model != "unified":
            raise AttributeError(
                f"{model} is not a supported dielectric rough surface model."
            )
        self._model = model
        # the UNIFIED model has delta components a shadow ray cannot reach and a
        # walk without a closed form density, so NEE is unavailable there
        self._supportNEE = bool(supportNEE) and model != "unified"

        # Worst-case RNG draws for prepareSurface. The facet loop runs 21 times,
        # the last one being the fall-back, which draws no facet. The normal
        # sampler draws 2, except "gaussian" (and hence "unified") which
        # rejection-samples the facet angle (up to 4 attempts x 2 draws) plus 1
        # for the azimuth.
        sampler = 9 if model in ("gaussian", "unified") else 2
        if model == "unified":
            #   2 (pre-sampled diffuse direction)
            # + 20 * (normal + Fresnel coin + lobe choice)
            # + 2 (the fall-back still draws coin and lobe)
            prepare = 2 + 20 * (sampler + 2) + 2
        else:
            # one uniform per iteration decides channel and acceptance at once
            prepare = 20 * (sampler + 1) + 1
        draws = SurfaceRNGDraws(
            prepareSurface=prepare,
            sampleSurfaceInteraction=0,
            processSurfaceTargetHit=0,
            # the NEE proposal draws one facet plus a branch decision
            sampleSurfaceScattering=sampler + 1 if self._supportNEE else 0,
        )

        # the prob_* lobe weights are optional and thus not listed as required
        # material properties. All models parametrise the roughness through
        # `roughness_parameter`.
        requiredMaterialProperties = {"roughness_parameter"}
        super().__init__(
            rngDraws=draws,
            supportsMIS=self._supportNEE,
            requiredMediumProperties={"refractive_index"},
            requiredMaterialProperties=requiredMaterialProperties,
        )

    @property
    def model(self) -> str:
        return self._model

    @property
    def supportNEE(self) -> bool:
        """Whether this surface can be reached by next event estimation"""
        return self._supportNEE

    def _sourceCode(self, mode: str) -> str:
        # the micro-facet code is prepended to the shared surface implementation
        # and compiled together
        if self.model == "unified":
            # the UNIFIED model is always gaussian and marks itself specular, so
            # it needs neither a model choice nor the NEE config header
            return "\n".join([
                createPreamble(SURFACE_UNIFIED_TRANSMISSION=True),
                loadShader("surface/microfacet_models/gaussian.glsl"),
                loadShader(f"surface/unified/{mode}.glsl"),
            ])
        return "\n".join([
            createPreamble(SURFACE_ROUGH_NO_NEE=not self._supportNEE),
            # resolves ROUGH_SURFACE_NEE; has to come before the model, which
            # keys its NEE only parts off it
            loadShader("surface/rough_nee_config.glsl"),
            loadShader(f"surface/microfacet_models/{self.model}.glsl"),
            loadShader(f"surface/dielectric_rough/{mode}.glsl"),
        ])

    @property
    def backwardSourceCode(self) -> str:
        return self._sourceCode("backward")

    @property
    def forwardSourceCode(self) -> str:
        return self._sourceCode("forward")

    def derivedMaterialProperties(
        self, material, media: Mapping[str, object]
    ) -> Mapping[str, Property]:
        return _roughDerivedProperties(
            self.model,
            self._supportNEE,
            material,
            media,
            transmission=True,
        )

    def finalizeMaterials(self, store, materialIndices: list[int]) -> None:
        if not self._supportNEE:
            return
        # both directions share the axis; the outward one is evaluated at 1/e
        fillNormalizationTables(
            store,
            self.forwardSourceCode,
            materialIndices,
            [("microfacet_z_inward", False), ("microfacet_z_outward", True)],
        )

    @classmethod
    def load(cls, file: Traversable) -> DielectricRoughSurface:
        with file.open() as f:
            config = json.load(f)
        return DielectricRoughSurface(
            model=config["model"],
            supportNEE=config.get("supportNEE", True),
        )

    def save(self, file) -> None:
        config = {
            "model": self.model,
            "supportNEE": self._supportNEE,
        }
        with file.open("w") as f:
            json.dump(config, f)

    def __eq__(self, value: object) -> bool:
        return (
            type(value) == DielectricRoughSurface
            and value.model == self.model
            and value.supportNEE == self.supportNEE
        )

    def __hash__(self) -> int:
        return hash(self.name) ^ hash(self.model) ^ hash(self.supportNEE)


class MetallicSurface(SurfaceModel, name="metallic"):
    """
    Models absorption and specular reflection on a metallic surface.
    Reflectivity can either be provided directly or is calculated using the
    complex refractive index of the metal.

    If `absorb` is set to `True`, rays may be absorbed to stop tracing.
    Otherwise they will always reflect attenuated by the surface's reflectance
    if it has non-zero reflectance.
    """

    def __init__(self, *, absorb: bool = False) -> None:
        draws = SurfaceRNGDraws(
            prepareSurface=1,
            sampleSurfaceInteraction=0,
            processSurfaceTargetHit=0,
        )
        super().__init__(
            rngDraws=draws,
            requiredMaterialProperties={"reflectivity"},
            requiredMediumProperties={"refractive_index", "imag_refractive_index"},
        )
        self._absorb = absorb

    @property
    def isAbsorbing(self) -> bool:
        """Whether ray may be absorbed to stop tracing"""
        return self._absorb

    @property
    def forwardSourceCode(self) -> str:
        preamble = createPreamble(SURFACE_ABSORB_RAY=self.isAbsorbing)
        return loadShader("surface/metallic/forward.glsl", preamble)

    @property
    def backwardSourceCode(self) -> str:
        preamble = createPreamble(SURFACE_ABSORB_RAY=self.isAbsorbing)
        return loadShader("surface/metallic/backward.glsl", preamble)

    @classmethod
    def load(cls, file: Traversable) -> MetallicSurface:
        with file.open() as f:
            config = json.load(f)
        return MetallicSurface(absorb=config["absorb"])

    def save(self, file) -> None:
        config = {"absorb": self.isAbsorbing}
        with file.open("w") as f:
            json.dump(config, f)

    def __eq__(self, value: object) -> bool:
        return type(value) == MetallicSurface and value.isAbsorbing == self.isAbsorbing

    def __hash__(self) -> int:
        return hash(self.name) ^ hash(self.isAbsorbing)


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


class ThinDielectricSurface(SurfaceModel, name="thin_dielectric"):
    """
    Models transmission and reflection through a thin dielectric layer between
    two other dielectric media. Optionally includes interference effects by
    setting `interference` to `True`.
    """

    def __init__(self, *, interference=False) -> None:
        draws = SurfaceRNGDraws(
            prepareSurface=1,
            sampleSurfaceInteraction=0,
            processSurfaceTargetHit=0,
        )
        super().__init__(
            rngDraws=draws,
            requiredMediumProperties={"refractive_index"},
            requiredMaterialProperties={"layer_thickness"},
        )
        self._interference = interference

    @property
    def isInterfering(self) -> bool:
        """Whether interference effects are included"""
        return self._interference

    @property
    def forwardSourceCode(self) -> str:
        preamble = createPreamble(
            SURFACE_THIN_LAYER_INCLUDE_INTERFERENCE=self.isInterfering
        )
        shader = loadShader("surface/thin/dielectric/forward.glsl")
        return preamble + shader

    @property
    def backwardSourceCode(self) -> str:
        preamble = createPreamble(
            SURFACE_THIN_LAYER_INCLUDE_INTERFERENCE=self.isInterfering
        )
        shader = loadShader("surface/thin/dielectric/backward.glsl")
        return preamble + shader

    @classmethod
    def load(cls, file: Traversable) -> ThinDielectricSurface:
        with file.open() as f:
            config = json.load(f)
        return ThinDielectricSurface(interference=config["interference"])

    def save(self, file) -> None:
        config = {"interference": self.isInterfering}
        with file.open("w") as f:
            json.dump(config, f)

    def __eq__(self, value: object) -> bool:
        return (
            type(value) == ThinDielectricSurface
            and value.isInterfering == self.isInterfering
        )

    def __hash__(self) -> int:
        return hash(self.name) ^ hash(self.isInterfering)


class ThinMetallicSurface(SurfaceModel, name="thin_metallic"):
    """
    Models transmission, reflection and absorption in a thin metallic layer
    between two dielectric media including interference effects within the thin
    layer.

    When used as detector surface, will report absorbed photons as detected.
    If `absorb` is set to `True`, rays may be absorbed to stop tracing.
    """

    def __init__(self, *, absorb=True) -> None:
        draws = SurfaceRNGDraws(
            prepareSurface=1,
            sampleSurfaceInteraction=1,
            processSurfaceTargetHit=0,
        )
        super().__init__(
            rngDraws=draws,
            requiredMediumProperties={"imag_refractive_index", "refractive_index"},
            requiredMaterialProperties={"layer_medium_idx", "layer_thickness"},
        )
        self._absorb = absorb

    @property
    def isAbsorbing(self) -> bool:
        """Whether rays may be absorbed in the thin layer"""
        return self._absorb

    @property
    def forwardSourceCode(self) -> str:
        preamble = createPreamble(SURFACE_ABSORB_RAY=self.isAbsorbing)
        shader = loadShader("surface/thin/metallic/forward.glsl")
        return preamble + shader

    @property
    def backwardSourceCode(self) -> str:
        preamble = createPreamble(SURFACE_ABSORB_RAY=self.isAbsorbing)
        shader = loadShader("surface/thin/metallic/backward.glsl")
        return preamble + shader

    @classmethod
    def load(cls, file: Traversable) -> ThinMetallicSurface:
        with file.open() as f:
            config = json.load(f)
        return ThinMetallicSurface(absorb=config["absorb"])

    def save(self, file) -> None:
        config = {"absorb": self.isAbsorbing}
        with file.open("w") as f:
            json.dump(config, f)

    def __eq__(self, value: object) -> bool:
        return (
            type(value) == ThinMetallicSurface and value.isAbsorbing == self.isAbsorbing
        )

    def __hash__(self) -> int:
        return hash(self.name) ^ hash(self.isAbsorbing)
