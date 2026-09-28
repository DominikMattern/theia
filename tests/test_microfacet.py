"""
Independent reference for the micro-facet normalization tables.

theia builds those tables on the device from the surface model's own functions
(`eval_microfacet_density` and `roughFacetAcceptance`), so the shipped code
contains the facet density and the acceptance criteria exactly once - a table can
therefore not drift away from the sampling it normalizes.

The price is that nothing checks the tables any more. This module pays it back by
reimplementing the quadrature in numpy: a deliberate second implementation,
written from the model definitions rather than transcribed from the shader, whose
only job is to disagree if either side changes.
"""

from __future__ import annotations

import hephaistos as hp
import numpy as np
import pytest

from ctypes import Structure, c_float, c_uint32

from theia.compiler import compileShader, createPreamble, loadShader
from theia.lookup import Table
from theia.material import Material, MaterialStore, Medium
from theia.microfacet import (
    MICROFACET_MODELS,
    _ETA_SAMPLES,
    _MU_SAMPLES,
    gaussianNormalization,
    muRange,
)
from theia.property import FloatProperty, TableProperty
import theia.surface

N_THETA, N_PHI = 256, 128
THETA_WARP = 3.0
"""
Quadrature of the facet integral, mirroring the constants in
shader/surface/microfacet/normalization.glsl. The polar angle is warped because
the facet density is peaked at small angles; in the warped variable the integrand
vanishes like u**5 and is smooth across the whole roughness range.
"""


# --------------------------------------------------------------------------- #
# numpy reference
# --------------------------------------------------------------------------- #


def masking(cos_n, alpha):
    """Smith masking term of the Trowbridge-Reitz model"""
    cos_n = np.abs(cos_n)
    tan2 = np.clip(1.0 - cos_n**2, 0.0, None) / np.clip(cos_n**2, 1e-30, None)
    lam = (np.sqrt(1.0 + alpha * alpha * tan2) - 1.0) / 2.0
    return np.where(cos_n < 1e-3, 0.0, 1.0 / (1.0 + lam))


def facetDensity(model, rough, cos_h, cos_i, mu_i):
    """Density with which the model draws a facet, in solid angle"""
    cos_h = np.clip(cos_h, 0.0, 1.0)
    c2 = np.clip(cos_h**2, 1e-30, None)
    tan2 = (1.0 - c2) / c2
    a2 = rough * rough

    if model == "beckmann":
        return np.exp(-tan2 / a2) / (np.pi * a2 * c2 * cos_h)
    if model == "trowbridge_reitz":
        d = a2 + tan2
        return a2 / (np.pi * c2 * cos_h * d * d)
    if model == "gaussian":
        theta = np.arccos(np.minimum(cos_h, 1.0))
        return np.exp(-(theta**2) / (2.0 * rough**2)) / (
            2.0 * np.pi * gaussianNormalization(rough)
        )
    if model == "trowbridge_reitz_shadowed":
        d = a2 + tan2
        D = a2 / (np.pi * c2 * c2 * d * d)
        # distribution of visible normals
        return np.where(
            cos_i > 0.0,
            masking(mu_i, rough) * cos_i * D / np.maximum(mu_i, 1e-30),
            0.0,
        )
    raise ValueError(f"unknown micro-facet model: {model}")


def acceptProb(model, rough, cos_out, cos_i):
    """Probability that the model accepts a facet for the given outgoing direction"""
    if model == "trowbridge_reitz_shadowed":
        return masking(cos_out, rough)
    return np.where(cos_i > 0.0, 1.0, 0.0)  # facet hit from the front


def fresnel(c, eta):
    """Unpolarized Fresnel reflectance for eta = n_i/n_o"""
    c = np.abs(c)
    sin_o = np.sqrt(np.clip(1.0 - c * c, 0.0, None)) * eta
    cos_o = np.sqrt(np.clip(1.0 - sin_o * sin_o, 0.0, None))
    r_s = (eta * c - cos_o) / (eta * c + cos_o)
    r_p = (c - eta * cos_o) / (c + eta * cos_o)
    return 0.5 * (r_s**2 + r_p**2)


def acceptanceRate(model, rough, mu, eta):
    """
    Acceptance rate of the facet loop, i.e. the `Z` the surface model divides its
    BSDF by. `eta is None` gives the reflection only rate a metal needs.

    Evaluated for every entry of `mu` and, if given, every entry of `eta`.
    """
    u = (np.arange(N_THETA) + 0.5) / N_THETA
    theta = (np.pi / 2) * u**THETA_WARP
    dTheta = (np.pi / 2) * THETA_WARP * u ** (THETA_WARP - 1) / N_THETA
    phi = (np.arange(N_PHI) + 0.5) * np.pi / N_PHI
    # factor 2 for the mirrored azimuth half
    dA = (2.0 * np.sin(theta) * dTheta)[:, None] * (np.pi / N_PHI)

    # shape (mu, theta, phi)
    m = np.asarray(mu, dtype=np.float64)[:, None, None]
    si = np.sqrt(np.clip(1.0 - m**2, 0.0, None))
    cos_h = np.cos(theta)[None, :, None]
    sin_h = np.sin(theta)[None, :, None]
    cos_p = np.cos(phi)[None, None, :]

    # c = w_i.h with w_i = (sin_i, 0, mu) and h = (sin_h cos_p, sin_h sin_p, cos_h)
    c = si * sin_h * cos_p + m * cos_h
    shape = c.shape
    D = facetDensity(
        model, rough, np.broadcast_to(cos_h, shape), c, np.broadcast_to(m, shape)
    )

    # reflection channel
    cos_r = 2.0 * c * cos_h - m
    a_r = np.where(cos_r > 0.0, acceptProb(model, rough, cos_r, c), 0.0)
    if eta is None:
        return np.einsum("mtp,tp->m", D * a_r, dA)

    etas = np.atleast_1d(np.asarray(eta, dtype=np.float64))
    Dr = D * a_r
    one_c2 = 1.0 - c * c
    out = np.empty(np.shape(mu) + etas.shape)
    for i, e in enumerate(etas):
        F = fresnel(c, e)
        # refraction channel; c_t == 0 marks total internal reflection, F == 1 there
        c_t = np.sqrt(np.clip(1.0 - e * e * one_c2, 0.0, None))
        cos_t = (e * c - c_t) * cos_h - e * m
        a_t = np.where(cos_t < 0.0, acceptProb(model, rough, cos_t, c), 0.0)
        integrand = F * Dr + (1.0 - F) * (D * a_t)
        out[..., i] = np.einsum("mtp,tp->m", integrand, dA)
    return out if np.ndim(eta) else out[..., 0]


# --------------------------------------------------------------------------- #
# device side
# --------------------------------------------------------------------------- #


class ProbeParams(Structure):
    _fields_ = [
        ("materialIdx", c_uint32),
        ("slot", c_uint32),
        ("nu", c_uint32),
        ("nv", c_uint32),
    ]


def _medium(name, n):
    return Medium(
        name=name,
        wavelengthRange=(300.0, 700.0),
        properties={
            "refractive_index": TableProperty(
                Table(np.full(32, n, np.float32), (300.0, 700.0))
            )
        },
    )


def _readTable(store, slot, nu, nv):
    """Reads a normalization table back through the regular look up path"""
    program = hp.Program(
        compileShader("microfacet.normalization.test.glsl", headers=store.header)
    )
    n = nu * max(nv, 1)
    out, buf = hp.Tensor(n * 4), hp.Buffer(n * 4)
    store.bindParams(program)
    program.bindParams(OutBuf=out)
    params = hp.StructureTensor(ProbeParams, mapped=True)
    view = ProbeParams.from_address(params.memory)
    view.materialIdx = 0
    view.slot = store.materials.slots[slot]
    view.nu, view.nv = nu, nv
    program.bindParams(ProbeParams=params)
    hp.execute(program.dispatch(-(n // -64)))
    hp.execute(hp.retrieveTensor(out, buf))
    got = np.ctypeslib.as_array((c_float * n).from_address(buf.address)).copy()
    return got.reshape(nu, nv) if nv else got


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #

N_INSIDE, N_OUTSIDE = 1.33, 1.50


@pytest.mark.parametrize("model", MICROFACET_MODELS)
@pytest.mark.parametrize("rough", [0.05, 0.15, 0.4])
def test_facetDensityNormalized(model, rough):
    """
    The facet density has to integrate to one over the upper hemisphere. This
    guards the reference itself, without which the comparisons below would only
    show that two implementations agree on being wrong.
    """
    nt, npp = 4000, 720
    theta = (np.arange(nt) + 0.5) * (np.pi / 2) / nt
    phi = (np.arange(npp) + 0.5) * 2 * np.pi / npp
    T, P = np.meshgrid(theta, phi, indexing="ij")
    # incidence along the normal keeps the visible normal distribution simple
    cos_i = np.cos(T)
    D = facetDensity(model, rough, np.cos(T), cos_i, np.ones_like(T))
    dA = np.sin(T) * (np.pi / 2 / nt) * (2 * np.pi / npp)
    assert float(np.sum(D * dA)) == pytest.approx(1.0, abs=1e-4)


def test_gaussianNormalization():
    """Z_sigma against an independent quadrature"""
    for sigma in (0.02, 0.05, 0.1, 0.2, 0.4, 1.0):
        t = np.linspace(0.0, np.pi / 2, 200001)
        ref = np.trapezoid(np.sin(t) * np.exp(-t * t / (2 * sigma**2)), t)
        assert gaussianNormalization(sigma) == pytest.approx(ref, rel=1e-5)


@pytest.mark.parametrize(
    "surfaceType", [theia.surface.DielectricRoughSurface, theia.surface.MetalRoughSurface]
)
@pytest.mark.parametrize("model", MICROFACET_MODELS + ["unified"])
@pytest.mark.parametrize("supportNEE", [True, False])
def test_compiles(surfaceType, model, supportNEE):
    """
    Every model has to compile in every configuration.

    What a model provides only for NEE is behind a macro guard, and so are the
    material properties feeding it - `gaussian` needs a precomputed normalization
    for its facet density. Getting one of the two wrong shows up here and nowhere
    else, since the configurations without NEE never reach the tests above.
    """
    surface = surfaceType(model=model, supportNEE=supportNEE)
    mat = Material(
        "mat", _medium("in", N_INSIDE), _medium("out", N_OUTSIDE), surface,
        properties={"roughness_parameter": FloatProperty(0.15)},
    )
    store = MaterialStore([mat])

    # particle rays are one of the reasons NEE gets disabled, so cover them too
    for particle in (False, True):
        compileShader(
            "surface/sample/interaction.glsl",
            createPreamble(
                SAMPLE_FORWARD=True,
                RAY_FIELD_COUNT=16,
                RAY_PARTICLE=particle,
                LIGHT_SOURCE_EMIT_PARTICLE=particle,
            ),
            {
                "ray.glsl": loadShader("ray/unpolarized.glsl"),
                "rng.glsl": loadShader("random/philox.glsl"),
                "photon.glsl": loadShader("wavelengthsource/uniform.glsl"),
                "source.glsl": loadShader("lightsource/pencil.glsl"),
                "surface.glsl": surface.forwardSourceCode,
                **store.header,
            },
        )


@pytest.mark.parametrize("model", MICROFACET_MODELS)
@pytest.mark.parametrize("rough", [0.05, 0.15, 0.4])
def test_metalNormalizationTable(model, rough):
    """The device built table of a rough metal against the numpy reference"""
    surface = theia.surface.MetalRoughSurface(model=model)
    mat = Material(
        "mat", _medium("in", N_INSIDE), _medium("out", N_OUTSIDE), surface,
        properties={"roughness_parameter": FloatProperty(rough)},
    )
    store = MaterialStore([mat])

    got = _readTable(store, "microfacet_z_refl", _MU_SAMPLES, 0)
    mu = np.linspace(*muRange(), _MU_SAMPLES)
    assert got == pytest.approx(acceptanceRate(model, rough, mu, None), abs=1e-3)


@pytest.mark.parametrize("model", MICROFACET_MODELS)
@pytest.mark.parametrize("rough", [0.05, 0.15, 0.4])
def test_dielectricNormalizationTables(model, rough):
    """
    The device built tables of a rough dielectric against the numpy reference.

    Both traversal directions share one axis, the inward ratio
    `n_outside/n_inside`; the outward table is evaluated at its reciprocal.
    """
    surface = theia.surface.DielectricRoughSurface(model=model)
    mat = Material(
        "mat", _medium("in", N_INSIDE), _medium("out", N_OUTSIDE), surface,
        properties={"roughness_parameter": FloatProperty(rough)},
    )
    store = MaterialStore([mat])

    mu = np.linspace(*muRange(), _MU_SAMPLES)
    # same axis as _etaRange derives for constant indices
    ratio = N_OUTSIDE / N_INSIDE
    eta = np.linspace(ratio * 0.99, ratio * 1.01, _ETA_SAMPLES)

    for slot, etaEval in (
        ("microfacet_z_inward", eta),
        ("microfacet_z_outward", 1.0 / eta),
    ):
        got = _readTable(store, slot, _MU_SAMPLES, _ETA_SAMPLES)
        ref = acceptanceRate(model, rough, mu, etaEval)
        assert got == pytest.approx(ref, abs=1e-3)
