"""
Normalization tables for the rough micro-facet surface models.

`prepareSurface()` in `shader/surface/{dielectric,metal}_rough/template.glsl`
draws micro-facets until one is accepted, so the density of the accepted facet is
the sampling density renormalized by the acceptance rate. That rate has no closed
form, but NEE needs it in absolute terms: it is the `Z` in the BSDF the surface
model implicitly defines.

It is computed on the *device*, by `surface/microfacet/normalization.glsl`, from
the surface model's own functions - `eval_microfacet_density` and
`roughFacetAcceptance`, the latter being the one the facet loop itself uses. A
table can therefore not drift away from the sampling it normalizes, and the facet
density, the Fresnel equations and the acceptance criteria exist exactly once.

This module only reserves the storage (`emptyNormalizationTable`, called from
`SurfaceModel.derivedMaterialProperties`) and drives that computation afterwards
(`fillNormalizationTables`, called from `SurfaceModel.finalizeMaterials`). The
split is what resolves the ordering: the acceptance rate depends on the roughness
and the refractive indices, which only become readable once the material table is
complete - so the shader writes the samples in place into memory the property
already reserved.

An independent numpy reimplementation of the quadrature lives in
`tests/test_microfacet.py`. Deliberately duplicated there and nowhere else, its
only job is to disagree if either side changes.

The dielectric table is indexed by the *inward* ratio `n_outside / n_inside`,
taken at equal wavelengths. Both traversal directions share that axis - a ray
going inward sees it as its eta, one going outward its reciprocal - so one table
per direction covers everything. Pairing the wavelengths keeps that range narrow,
and it only straddles `eta = 1` if the two media genuinely cross in refractive
index; there the interface is optically absent and the acceptance rate spikes,
which no reasonable grid resolves.
"""

from __future__ import annotations

import numpy as np

from theia.lookup import Table

import hephaistos as hp

from collections.abc import Sequence
from ctypes import Structure, c_uint32
from functools import lru_cache

__all__ = [
    "MICROFACET_MODELS",
    "emptyNormalizationTable",
    "fillNormalizationTables",
    "gaussianNormalization",
    "muRange",
]


def __dir__():
    return __all__


MICROFACET_MODELS: list[str] = [
    "beckmann",
    "trowbridge_reitz",
    "trowbridge_reitz_shadowed",
    "gaussian",
]
"""Micro-facet models the rough surfaces can be parametrized with"""


def gaussianNormalization(sigma: float) -> float:
    """
    Returns ``Z_sigma = int_0^(pi/2) sin(t) exp(-t^2 / (2 sigma^2)) dt``, the
    normalization of the Gaussian facet angle distribution. Has no closed form
    and is needed to evaluate the model's facet density (and thus its BSDF).

    The integrand has its mass around ``t ~ sigma``, so the grid is truncated to
    a few sigma rather than spread over the whole domain: at ``sigma = 0.02`` a
    uniform grid over ``[0, pi/2]`` would place barely a hundred of its points
    inside the peak. What is cut off is suppressed by ``exp(-50)``.
    """
    n = 4096
    hi = min(np.pi / 2, 10.0 * sigma)
    t = (np.arange(n) + 0.5) * hi / n
    return float(np.sum(np.sin(t) * np.exp(-t * t / (2.0 * sigma**2))) * hi / n)


# --------------------------------------------------------------------------- #
# Device side table construction
# --------------------------------------------------------------------------- #

_MU_SAMPLES = 64
"""Default samples of the incidence cosine axis"""
_ETA_SAMPLES = 32
"""Default samples of the refractive index ratio axis"""


def muRange(nMu: int = _MU_SAMPLES) -> tuple[float, float]:
    """
    Range the incidence cosine axis covers. ``mu_i = 0`` is a grazing hit with
    vanishing acceptance, so the grid starts half a step above it.
    """
    return (1.0 / (2 * nMu), 1.0)


def emptyNormalizationTable(
    etaRange: tuple[float, float] | None = None,
    *,
    nMu: int = _MU_SAMPLES,
    nEta: int = _ETA_SAMPLES,
) -> Table:
    """
    Table with the right shape and axes but without data. `fillNormalizationTables`
    computes the samples on the device once the material table exists - which it
    has to, since the acceptance rate depends on properties stored in it.
    """
    if etaRange is None:
        return Table(np.zeros(nMu, np.float32), muRange(nMu))
    return Table(np.zeros((nMu, nEta), np.float32), muRange(nMu), etaRange)


class _NormalizationParams(Structure):
    _fields_ = [
        ("materialIdx", c_uint32),
        ("slot", c_uint32),
        ("reciprocal", c_uint32),
    ]


@lru_cache(maxsize=8)
def _normalizationCode(surfaceSource: str, slotHeader: str) -> bytes:
    """
    Compiles the normalization shader for one surface model.

    Only the byte code is cached, not a `Program`: the latter holds bindings to
    the store's tensors and must not outlive them.
    """
    from theia.compiler import compileShader, createPreamble, loadShader

    code = compileShader(
        "surface/microfacet/normalization.glsl",
        createPreamble(),
        {
            "ray.glsl": loadShader("ray/unpolarized.glsl"),
            "rng.glsl": loadShader("random/philox.glsl"),
            "surface.glsl": surfaceSource,
            "slots.glsl": slotHeader,
        },
    )
    return code


def fillNormalizationTables(
    store,
    surfaceSource: str,
    materialIndices: Sequence[int],
    slots: Sequence[tuple[str, bool]],
    *,
    nMu: int = _MU_SAMPLES,
    nEta: int = _ETA_SAMPLES,
) -> None:
    """
    Fills the acceptance rate tables of a rough surface model on the device.

    The tables live in the memory `TableProperty` reserved inside the material
    table, and the shader writes them in place through a writable alias. That
    resolves the ordering problem: the acceptance rate is a function of the
    roughness and the refractive indices, which only become readable once the
    material table is complete.

    Parameters
    ----------
    store: MaterialStore
        Store holding the tables. Must be fully constructed.
    surfaceSource: str
        Shader source of the surface model, providing `roughFacetAcceptance`.
    materialIndices: [int]
        Indices of the materials whose tables to fill.
    slots: [(str, bool)]
        Slot name and whether it holds the reciprocal direction, i.e. the table
        is evaluated at `1/e` while still being indexed by `e`.
    """
    program = hp.Program(_normalizationCode(surfaceSource, store.header["slots.glsl"]))
    store.bindParams(program)
    params = hp.StructureTensor(_NormalizationParams, mapped=True)
    view = _NormalizationParams.from_address(params.memory)
    program.bindParams(NormalizationParams=params)

    slotIndices = store.materials.slots
    # one invocation per padded table entry
    n2d = (nMu + 2 * Table.PADDING) * (nEta + 2 * Table.PADDING)
    n1d = nMu + 2 * Table.PADDING
    for materialIdx in materialIndices:
        for name, reciprocal in slots:
            count = n1d if name.endswith("_refl") else n2d
            view.materialIdx = materialIdx
            view.slot = slotIndices[name]
            view.reciprocal = int(reciprocal)
            hp.execute(program.dispatch(-(count // -64)))
