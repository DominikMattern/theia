import pytest

import hephaistos.pipeline as pl
import numpy as np
from scipy.stats import expon as scipy_expon, rv_continuous

from theia.camera import ConeCamera, PencilCamera
from theia.light import ConeLightSource, ConstWavelengthSource, PencilLightSource
from theia.material import Material, MaterialStore, VACUUM_IDX
from theia.model import BK7Model, PureWaterModel
from theia.property import TableProperty, FloatProperty
from theia.random import PhiloxRNG
from theia.ray import UnpolarizedRay
from theia.testing import SurfaceInteractionSampler, SurfaceScatteringSampler
from theia.trace import EventResultCode
import theia.units as u
import theia.surface


def reflect(i, n):
    ct = np.multiply(i, n[None, :]).sum(1)
    return i - 2.0 * ct[:, None] * n


def refract(i, n, ni, no):
    eta = ni / no
    ct = np.multiply(i, n[None, :]).sum(1)
    k = 1.0 - eta * eta * (1.0 - ct * ct)
    return eta[:, None] * i - (eta * ct + np.sqrt(k))[:, None] * n


def reflectance(i, n, ni, no):
    ci = np.abs(np.multiply(i, n[None, :]).sum(1))
    si = np.sqrt(np.clip(1.0 - np.square(ci), 0.0, 1.0))
    so = si * ni / no
    co = np.sqrt(np.clip(1.0 - np.square(so), 0.0, 1.0))
    rs = (ni * ci - no * co) / (ni * ci + no * co)
    rp = (no * ci - ni * co) / (no * ci + ni * co)
    return 0.5 * (rs * rs + rp * rp)


@pytest.mark.parametrize(
    "particle,camera,flags",
    [
        (True, False, "TR"),
        (True, False, "T"),
        (True, False, "R"),
        (True, False, "DR"),
        (False, True, "TR"),
        (False, True, "T"),
        (False, True, "R"),
        (False, False, "TR"),
        (False, False, "T"),
        (False, False, "R"),
        (False, False, "DR"),
    ],
)
def test_DielectricSurface(particle: bool, camera: bool, flags: str):
    N = 32 * 1024
    lam = 600.0 * u.nm
    direction = (0.8, 0.36, 0.48)
    normal = (-0.8, -0.36, -0.48)
    objectId = 10

    # create materials
    waterModel = PureWaterModel()
    water = waterModel.createMedium()
    surface = theia.surface.DielectricSurface()
    mat = Material("mat", None, water, surface, flags=flags)
    matStore = MaterialStore([mat])
    waterIdx = matStore.media["water"]

    # create pipeline
    ray = UnpolarizedRay(particle=particle)
    rng = PhiloxRNG(key=0xABBA)
    photons = ConstWavelengthSource(lam)
    if camera:
        source = ConeCamera(
            direction=direction,
            cosOpeningAngle=0.0,
            mediumIdx=waterIdx,
            objectId=objectId,
        )
    else:
        source = ConeLightSource(
            photons,
            direction=direction,
            cosOpeningAngle=0.0,
            mediumIdx=waterIdx,
            timeRange=(0.0, 0.0),
            emitParticles=particle,
        )
        photons = None
    sampler = SurfaceInteractionSampler(
        N,
        ray,
        surface,
        matStore,
        rng,
        source,
        photons,
        material="mat",
        surfaceNormal=normal,
        objectId=objectId,
        sampleTargetHit=not camera,
    )
    # run pipeline
    pl.runPipeline(sampler.collectStages())

    # check results
    result = sampler.queue.view(0)
    assert np.all(result["positionIn"] == 0.0)
    assert np.all(result["wavelengthIn"] == lam)
    assert np.all(result["wavelengthOut"] == lam)
    assert np.all(result["mediumIdxIn"] == waterIdx)
    if not camera:
        # only consider valid hits
        valid = result["hitSuccess"] == 1
        assert np.all(result["wavelengthHit"][valid] == lam)
        assert np.all(result["objectIdHit"][valid] == objectId)
    if flags == "D":
        # neither reflect nor transmit flag -> absorb
        assert np.all(result["hitResult"] == EventResultCode.RAY_ABSORBED)
        return
    absorbed = result["hitResult"] == EventResultCode.RAY_ABSORBED
    assert np.all(result["hitResult"][~absorbed] == EventResultCode.RAY_HIT)

    normal = np.array(normal)
    cosNrm = np.multiply(result["directionOut"], normal[None, :]).sum(1)
    trans = (cosNrm < 0.0) & ~absorbed
    refl = (cosNrm >= 0.0) & ~absorbed
    ni = waterModel.refractive_index(lam) * np.ones(N)
    no = np.ones(N)
    t = refract(result["directionIn"], normal, ni, no)
    r = reflect(result["directionIn"], normal)
    R = reflectance(result["directionIn"], normal, ni, no)
    if not particle:
        c = result["contribIn"]
    if camera:
        # in backward mode transmittance scales with a factor eta^2
        eta = ni / no
        c = np.copy(result["contribIn"])
        c[trans] = (c * eta * eta)[trans]

    # check we prevent self intersection
    cosPos = np.multiply(result["positionOut"], normal[None, :]).sum(-1)
    assert np.all(cosPos[trans] < 0.0)
    assert np.all(cosPos[refl] > 0.0)

    assert np.allclose(result["directionOut"][trans], t[trans], atol=1e-5)
    assert np.allclose(result["directionOut"][refl], r[refl], atol=1e-5)
    assert np.all(result["mediumIdxOut"][trans] == VACUUM_IDX)
    assert np.all(result["mediumIdxOut"][refl] == waterIdx)

    if flags == "R" or flags == "DR":
        assert trans.sum() == 0
        if particle:
            assert absorbed.sum() > 0
        else:
            cO = result["contribOut"]
            assert np.allclose(cO[~absorbed], (c * R)[~absorbed], atol=5e-4)
    if flags == "T":
        assert refl.sum() == 0
        assert absorbed.sum() > 0  # total internal reflection
        if not particle:
            cO = result["contribOut"]
            assert np.allclose(cO[~absorbed], (c * (1.0 - R))[~absorbed], atol=5e-4)
        if not particle and not camera:
            cH = result["contribHit"]
            assert np.allclose(cH[~absorbed], (c * (1.0 - R))[~absorbed], atol=5e-4)
    if flags == "TR":
        assert refl.sum() > 0
        assert trans.sum() > 0
        assert absorbed.sum() == 0
        if particle:
            # check particle is not both reflected and detected
            np.all((result["hitSuccess"] == 0) == refl)
        else:
            assert np.allclose(result["contribOut"], c)
        if not particle and not camera:
            cH = result["contribHit"]
            assert np.allclose(cH[~absorbed], (c * (1.0 - R))[~absorbed], atol=5e-5)


@pytest.mark.parametrize(
    "particle, camera", [(True, False), (False, True), (False, False)]
)
def test_AbsorbingSurface(particle: bool, camera: bool):
    N = 32 * 1024
    lam = 600.0 * u.nm
    direction = (0.8, 0.36, 0.48)
    normal = (-0.8, -0.36, -0.48)
    objectId = 10

    # create materials
    waterModel = PureWaterModel()
    water = waterModel.createMedium()
    surface = theia.surface.AbsorbingSurface()
    mat = Material("mat", None, water, surface)
    matStore = MaterialStore([mat])
    waterIdx = matStore.media["water"]

    # create pipeline
    ray = UnpolarizedRay(particle=particle)
    rng = PhiloxRNG(key=0xABBA)
    photons = ConstWavelengthSource(lam)
    if camera:
        source = ConeCamera(
            direction=direction,
            cosOpeningAngle=0.0,
            mediumIdx=waterIdx,
            objectId=objectId,
        )
    else:
        source = ConeLightSource(
            photons,
            direction=direction,
            cosOpeningAngle=0.0,
            mediumIdx=waterIdx,
            timeRange=(0.0, 0.0),
            emitParticles=particle,
        )
        photons = None
    sampler = SurfaceInteractionSampler(
        N,
        ray,
        surface,
        matStore,
        rng,
        source,
        photons,
        material="mat",
        surfaceNormal=normal,
        objectId=objectId,
        sampleTargetHit=not camera,
    )
    # run pipeline
    pl.runPipeline(sampler.collectStages())

    # check results
    result = sampler.queue.view(0)
    assert np.all(result["positionIn"] == 0.0)
    assert np.all(result["wavelengthIn"] == lam)
    assert np.all(result["wavelengthOut"] == lam)
    assert np.all(result["mediumIdxIn"] == waterIdx)
    assert np.all(result["hitResult"] == EventResultCode.RAY_ABSORBED)

    if not camera:
        assert np.all(result["hitSuccess"] == 1)
        assert np.all(result["positionHit"] == result["positionIn"])
        if not particle:
            assert np.all(result["contribHit"] == result["contribIn"])


@pytest.mark.parametrize(
    "particle, camera", [(True, False), (False, True), (False, False)]
)
def test_BorderSurface(particle: bool, camera: bool):
    N = 32 * 1024
    lam = 600.0 * u.nm
    direction = (0.8, 0.36, 0.48)
    normal = (-0.8, -0.36, -0.48)
    objectId = 10

    # create materials
    waterModel = PureWaterModel()
    water = waterModel.createMedium()
    surface = theia.surface.BorderSurface()
    mat = Material("mat", None, water, surface)
    matStore = MaterialStore([mat])
    waterIdx = matStore.media["water"]

    # create pipeline
    ray = UnpolarizedRay(particle=particle)
    rng = PhiloxRNG(key=0xABBA)
    photons = ConstWavelengthSource(lam)
    if camera:
        source = ConeCamera(
            direction=direction,
            cosOpeningAngle=0.0,
            mediumIdx=waterIdx,
            objectId=objectId,
        )
    else:
        source = ConeLightSource(
            photons,
            direction=direction,
            cosOpeningAngle=0.0,
            mediumIdx=waterIdx,
            timeRange=(0.0, 0.0),
            emitParticles=particle,
        )
        photons = None
    sampler = SurfaceInteractionSampler(
        N,
        ray,
        surface,
        matStore,
        rng,
        source,
        photons,
        material="mat",
        surfaceNormal=normal,
        objectId=objectId,
        sampleTargetHit=not camera,
    )
    # run pipeline
    pl.runPipeline(sampler.collectStages())

    # check results
    result = sampler.queue.view(0)
    assert np.all(result["positionIn"] == 0.0)
    assert np.all(result["wavelengthIn"] == lam)
    assert np.all(result["wavelengthOut"] == lam)
    assert np.all(result["mediumIdxIn"] == waterIdx)
    assert np.all(result["hitResult"] == EventResultCode.VOLUME_HIT)
    if not camera:
        assert np.all(result["hitSuccess"] == 0)

    # check we actually crossed the border (prevent self intersection)
    assert np.all(result["mediumIdxOut"] == VACUUM_IDX)
    normal = np.array(normal)
    cosPos = np.multiply(result["positionOut"], normal[None, :]).sum(-1)
    assert np.all(cosPos < 0.0)


@pytest.mark.parametrize(
    "particle,camera,flags",
    [
        (True, False, "TR"),
        (True, False, "T"),
        (True, False, "R"),
        (True, False, "DR"),
        (False, True, "TR"),
        (False, True, "T"),
        (False, True, "R"),
        (False, False, "TR"),
        (False, False, "T"),
        (False, False, "R"),
        (False, False, "DR"),
    ],
)
def test_LambertianReflectingSurface(particle: bool, camera: bool, flags: str):
    N = 32 * 1024
    lam = 600.0 * u.nm
    direction = (0.8, 0.36, 0.48)
    normal = (-0.8, -0.36, -0.48)
    objectId = 10
    reflectivity = 0.8

    # create materials
    waterModel = PureWaterModel()
    water = waterModel.createMedium()
    surface = theia.surface.LambertianReflectingSurface()
    probs = {"reflectivity": TableProperty.createConstTable(reflectivity)}
    mat = Material("mat", None, water, surface, flags=flags, properties=probs)
    matStore = MaterialStore([mat])
    waterIdx = matStore.media["water"]

    # create pipeline
    ray = UnpolarizedRay(particle=particle)
    rng = PhiloxRNG(key=0xABBA)
    photons = ConstWavelengthSource(lam)
    if camera:
        source = ConeCamera(
            direction=direction,
            cosOpeningAngle=0.0,
            mediumIdx=waterIdx,
            objectId=objectId,
        )
    else:
        source = ConeLightSource(
            photons,
            direction=direction,
            cosOpeningAngle=0.0,
            mediumIdx=waterIdx,
            timeRange=(0.0, 0.0),
            emitParticles=particle,
        )
        photons = None
    sampler = SurfaceInteractionSampler(
        N,
        ray,
        surface,
        matStore,
        rng,
        source,
        photons,
        material="mat",
        surfaceNormal=normal,
        objectId=objectId,
        sampleTargetHit=not camera,
    )
    # run pipeline
    pl.runPipeline(sampler.collectStages())

    # check results
    result = sampler.queue.view(0)
    assert np.all(result["positionIn"] == 0.0)
    assert np.all(result["wavelengthIn"] == lam)
    assert np.all(result["wavelengthOut"] == lam)
    assert np.all(result["mediumIdxIn"] == waterIdx)
    if not camera:
        # only consider valid hits
        valid = result["hitSuccess"] == 1
        assert np.all(result["wavelengthHit"][valid] == lam)
        assert np.all(result["objectIdHit"][valid] == objectId)
    canReflect = flags != "D" and flags != "T"
    if not canReflect:
        # no reflection -> absorb everything
        assert np.all(result["hitResult"] == EventResultCode.RAY_ABSORBED)
        return
    absorbed = result["hitResult"] == EventResultCode.RAY_ABSORBED
    assert np.all(result["hitResult"][~absorbed] == EventResultCode.RAY_HIT)

    # check contribution
    if particle:
        assert np.all(result["hitSuccess"][absorbed] == 1)
        assert np.all(result["hitSuccess"][~absorbed] == 0)
        assert abs(1.0 - absorbed.sum() / N - reflectivity) < 0.01  # noisy MC estimate
    else:
        if not camera:
            assert np.all(result["hitSuccess"] == 1)
            assert np.allclose(
                result["contribHit"], result["contribIn"] * (1.0 - reflectivity)
            )
        assert absorbed.sum() == 0
        assert np.allclose(
            result["contribIn"][~absorbed] * reflectivity,
            result["contribOut"][~absorbed],
        )

    # check we prevent self intersection
    normal = np.array(normal)
    cosPos = np.multiply(result["positionOut"], normal[None, :]).sum(-1)
    assert np.all(cosPos[~absorbed] > 0.0)  # all reflected
    assert np.all(result["mediumIdxOut"] == waterIdx)

    # check angular distribution of reflection
    # TODO: Be a bit more sophisticated here
    cosNrm = np.multiply(result["directionOut"][~absorbed], normal[None, :]).sum(1)
    assert cosNrm.min() > 0.0 and cosNrm.min() < 0.05
    assert cosNrm.max() > 0.95 and cosNrm.max() <= 1.0


@pytest.mark.parametrize(
    "camera,flags",
    [
        (True, "T"),
        (True, "R"),
        (True, "D"),
        (True, "TR"),
        (False, "T"),
        (False, "R"),
        (False, "D"),
        (False, "TR"),
    ],
)
def test_LambertianReflectingSurface_MIS(camera: bool, flags: str):
    N = 32 * 1024
    lam = 600.0 * u.nm
    direction = (0.8, 0.36, 0.48)
    normal = (-0.8, -0.36, -0.48)
    objectId = 10
    reflectivity = 0.8

    # create materials
    waterModel = PureWaterModel()
    water = waterModel.createMedium()
    surface = theia.surface.LambertianReflectingSurface()
    probs = {"reflectivity": TableProperty.createConstTable(reflectivity)}
    mat = Material("mat", None, water, surface, flags=flags, properties=probs)
    matStore = MaterialStore([mat])
    waterIdx = matStore.media["water"]

    # create pipeline
    ray = UnpolarizedRay()
    rng = PhiloxRNG(key=0xABBA)
    photons = ConstWavelengthSource(lam)
    if camera:
        source = ConeCamera(
            direction=direction,
            cosOpeningAngle=0.0,
            mediumIdx=waterIdx,
            objectId=objectId,
        )
    else:
        source = ConeLightSource(
            photons,
            direction=direction,
            cosOpeningAngle=0.0,
            mediumIdx=waterIdx,
            timeRange=(0.0, 0.0),
        )
        photons = None
    sampler = SurfaceScatteringSampler(
        N,
        ray,
        surface,
        matStore,
        rng,
        source,
        photons,
        material="mat",
        surfaceNormal=normal,
    )
    # run pipeline
    pl.runPipeline(sampler.collectStages())

    # check results
    result = sampler.queue.view(0)
    assert np.all(result["positionIn"] == 0.0)
    assert np.all(result["wavelengthIn"] == lam)
    assert np.all(result["wavelengthOut"] == lam)
    assert np.all(result["mediumIdxIn"] == waterIdx)

    assert np.allclose(result["probSampled"], result["probEval"])
    normal = np.array(normal)
    cosNrm = np.multiply(result["randomDir"], normal[None, :]).sum(-1)
    cosNrmSampled = np.multiply(result["sampledDir"], normal[None, :]).sum(-1)
    # TODO: more elaborate testing on the sampled distribution (also include phi...)
    assert cosNrmSampled.min() > 0.0 and cosNrmSampled.min() < 0.05
    assert cosNrmSampled.max() > 0.95 and cosNrmSampled.max() <= 1.0
    absorbed = result["scatterResult"] == EventResultCode.RAY_ABSORBED
    m = ~absorbed  # mask
    if flags == "T" or flags == "D":
        # cannot reflect -> no success
        assert np.all(absorbed)
        return  # nothing left to test
    assert np.all(~absorbed == (cosNrm > 0.0))
    assert np.all(absorbed == (cosNrm < 0.0))
    assert np.allclose(result["probRandom"][m], cosNrm[m] / np.pi)
    assert np.allclose(result["probRandom"][~m], 0.0)
    assert np.all(result["mediumIdxOut"][m] == waterIdx)
    expContrib = result["contribIn"][m] * reflectivity / np.pi
    assert np.allclose(result["contribOut"][m], expContrib)
    assert np.allclose(result["directionOut"][m], result["randomDir"][m])
    # check we prevent self intersection
    cosPos = np.multiply(result["positionOut"], normal[None, :]).sum(-1)
    assert np.all(cosPos[m] > 0.0)  # all reflected


def reflect_arr(i, n):
    ct = np.sum(i * n, axis=1)
    return i - 2.0 * ct[:, None] * n

def refract_arr(i, n, ni, no):
    eta = ni / no
    ct = np.sum(i * n, axis=1)
    k = 1.0 - eta * eta * (1.0 - ct * ct)
    return eta[:, None] * i - (eta * ct + np.sqrt(k))[:, None] * n

def reflectance_arr(i, n, ni, no):
    ci = np.abs(np.sum(i * n, axis=1))
    si = np.sqrt(np.clip(1.0 - np.square(ci), 0.0, 1.0))
    so = si * ni / no
    co = np.sqrt(np.clip(1.0 - np.square(so), 0.0, 1.0))
    rs = (ni * ci - no * co) / (ni * ci + no * co)
    rp = (no * ci - ni * co) / (no * ci + ni * co)
    return 0.5 * (rs * rs + rp * rp)

def sample_microfacet_normals_beckmann(normals, tangential_vector, direction, alpha, N):
    """
    Sample N microfacet normals from the Beckmann NDF.

    """
    tan2_theta = scipy_expon.rvs(scale=alpha**2, size=N)
    cos_theta = np.sqrt(1.0 / (1.0 + tan2_theta))
    sin_theta = np.sqrt(np.maximum(1.0 - cos_theta**2, 0.0))
    phi = np.random.uniform(0.0, 2.0 * np.pi, N)
    third_vectors = np.cross(normals, tangential_vector[None, :])
    return (
        cos_theta[:, None] * normals
        + (sin_theta * np.cos(phi))[:, None] * tangential_vector[None, :]
        + (sin_theta * np.sin(phi))[:, None] * third_vectors
    )


def sample_microfacet_normals_trowbridge_reitz(normals, tangential_vector, direction, alpha, N):
    """
    Sample N microfacet normals from the Trowbridge-Reitz (GGX) NDF.

    """
    u = np.random.uniform(0.0, 1.0, N)
    tan2_theta = alpha**2 * u / (1.0 - u)
    cos_theta = np.sqrt(1.0 / (1.0 + tan2_theta))
    sin_theta = np.sqrt(np.maximum(1.0 - cos_theta**2, 0.0))
    phi = np.random.uniform(0.0, 2.0 * np.pi, N)
    third_vectors = np.cross(normals, tangential_vector[None, :])
    return (
        cos_theta[:, None] * normals
        + (sin_theta * np.cos(phi))[:, None] * tangential_vector[None, :]
        + (sin_theta * np.sin(phi))[:, None] * third_vectors
    )


def _perpendicular_to_vector(v):
    """Unit vector perpendicular to v."""
    s = 1.0 if v[2] >= 0.0 else -1.0
    a = -1.0 / (s + v[2])
    b = v[0] * v[1] * a
    return np.array([b, s + v[1] * v[1] * a, -v[1]])


def _perpendicular_to_pair(a, b):
    """Normalize(cross(a, b)); falls back to _perpendicular_to_vector(a) if degenerate."""
    c = np.cross(a, b)
    length = np.linalg.norm(c)
    return c / length if length >= 1e-5 else _perpendicular_to_vector(a)


def _sample_unit_disk(N):
    """Concentric disk sampling. Returns (N, 2) array."""
    u = np.random.uniform(0.0, 1.0, (N, 2))
    rng = 2.0 * u - 1.0
    degenerate = (rng[:, 0] == 0.0) & (rng[:, 1] == 0.0)
    cond = np.abs(rng[:, 0]) > np.abs(rng[:, 1])
    safe_x = np.where(np.abs(rng[:, 0]) < 1e-30, 1.0, rng[:, 0])
    safe_y = np.where(np.abs(rng[:, 1]) < 1e-30, 1.0, rng[:, 1])
    r = np.where(cond, rng[:, 0], rng[:, 1])
    phi = np.where(
        cond,
        (np.pi / 4.0) * rng[:, 1] / safe_x,
        (np.pi / 2.0) - (np.pi / 4.0) * rng[:, 0] / safe_y,
    )
    px = np.where(degenerate, 0.0, r * np.cos(phi))
    py = np.where(degenerate, 0.0, r * np.sin(phi))
    return np.column_stack([px, py])


def _masking_function_trowbridge_reitz(cos_n, alpha):
    """Smith masking for Trowbridge-Reitz: 1 / (1 + Lambda)."""
    valid = cos_n >= 1e-3
    cos_n_safe = np.where(valid, cos_n, 1.0)
    tan2_n = np.maximum(1.0 - cos_n_safe**2, 0.0) / (cos_n_safe**2)
    lam = (np.sqrt(1.0 + alpha**2 * tan2_n) - 1.0) / 2.0
    return np.where(valid, 1.0 / (1.0 + lam), 0.0)


def sample_microfacet_normals_trowbridge_reitz_shadowed(normals, tangential_vector, direction, alpha, N):
    """
    Sample N microfacet normals from the VNDF for Trowbridge-Reitz.

    """
    # pencil beam: all normals identical
    normal = normals[0]  

    # Transformation matrix for local coordinate system. The surface normal is the new z-axis, and the
    # tangential component of the incoming ray is along the new y-axis.
    vx = _perpendicular_to_pair(normal, direction)
    vy = np.cross(normal, vx)
    trafo = np.column_stack([vx, vy, normal])

    cos_n = float(abs(np.dot(normal, direction)))
    sin_n = np.sqrt(max(1.0 - cos_n**2, 0.0))

    # transform surface normal to hemispherical configuration
    wh = np.array([0.0, alpha * sin_n, cos_n])
    wh_len = np.linalg.norm(wh)
    if wh_len > 1e-12:
        wh /= wh_len

    # Orthonormal basis for disk sampling
    T1 = np.array([1.0, 0.0, 0.0])
    T2 = np.cross(wh, T1) 

    # sample point on unit disk
    p = _sample_unit_disk(N)  # (N, 2)
    px, py = p[:, 0], p[:, 1]

    # warp hemispherical projection for visible normal sampling
    h = np.sqrt(np.maximum(1.0 - px**2, 0.0))
    t = (1.0 + wh[2]) / 2.0
    py_warped = h * (1.0 - t) + py * t

    # Reproject to hemisphere and transform normal to ellipsoid configuration
    pz = np.sqrt(np.maximum(0.0, 1.0 - px**2 - py_warped**2))

    nh = (
        px[:, None] * T1[None, :]
        + py_warped[:, None] * T2[None, :]
        + pz[:, None] * wh[None, :]
    )

    mn_local = np.column_stack([
        alpha * nh[:, 0],
        alpha * nh[:, 1],
        np.maximum(1e-6, nh[:, 2]),
    ])
    mn_local /= np.linalg.norm(mn_local, axis=1, keepdims=True)

    # transform from local to global coordinate system
    return mn_local @ trafo.T # (N, 3)


_ROUGH_DIELECTRIC_PARAMS = [
    (True, False, "TR", 0),
    (True, False, "TR", 30),
    (True, False, "TR", 85),
    (True, False, "T", 45),
    (True, False, "R", 45),
    (True, False, "DR", 45),
    (False, True, "TR", 0),
    (False, True, "TR", 30),
    (False, True, "TR", 85),
    (False, True, "T", 45),
    (False, True, "R", 45),
    (False, False, "TR", 0),
    (False, False, "TR", 30),
    (False, False, "TR", 85),
    (False, False, "T", 45),
    (False, False, "R", 45),
    (False, False, "DR", 45),
]


def _test_rough_dielectric_surface(surface, microfacet_sampler, particle, camera, flags, angle, alpha=0.05, masking_function=None):
    N = 64 * 1024
    lam = 600.0 * u.nm
    direction = (
        np.array((0.8, 0.36, 0.48)) * np.cos(np.deg2rad(angle))
        + np.array((-0.36, 0.8, 0)) * np.sin(np.deg2rad(angle)) / np.sqrt(481 / 625)
    )
    normal = np.array((-0.8, -0.36, -0.48)).astype(np.float32)
    tangential_vector = np.array((-0.36, 0.8, 0)) / np.sqrt(481 / 625)
    objectId = 10

    waterModel = PureWaterModel()
    water = waterModel.createMedium()
    roughness_prop = FloatProperty(alpha)
    mat = Material("mat", None, water, surface, flags=flags, properties={"roughness_parameter": roughness_prop})
    matStore = MaterialStore([mat])
    waterIdx = matStore.media["water"]

    ray = UnpolarizedRay(particle=particle)
    rng = PhiloxRNG(key=0xABBA)
    photons = ConstWavelengthSource(lam)
    if camera:
        source = PencilCamera(rayDirection=direction, mediumIdx=waterIdx, objectId=objectId)
    else:
        source = PencilLightSource(
            photons,
            direction=direction,
            mediumIdx=waterIdx,
            timeRange=(0.0, 0.0),
            emitParticles=particle,
        )
        photons = None
    sampler = SurfaceInteractionSampler(
        N, ray, surface, matStore, rng, source, photons,
        material="mat", surfaceNormal=normal, objectId=objectId,
        sampleTargetHit=not camera,
    )
    pl.runPipeline(sampler.collectStages())

    result = sampler.queue.view(0)
    assert np.all(result["positionIn"] == 0.0)
    assert np.all(result["wavelengthIn"] == lam)
    assert np.all(result["wavelengthOut"] == lam)
    assert np.all(result["mediumIdxIn"] == waterIdx)
    if not camera:
        assert np.allclose(result["directionIn"], direction, atol=1e-6)
        valid = result["hitSuccess"] == 1
        assert np.all(result["wavelengthHit"][valid] == lam)
        assert np.all(result["objectIdHit"][valid] == objectId)
    if flags == "D":
        assert np.all(result["hitResult"] == EventResultCode.RAY_ABSORBED)
        return
    absorbed = result["hitResult"] == EventResultCode.RAY_ABSORBED
    assert np.all(result["hitResult"][~absorbed] == EventResultCode.RAY_HIT)

    normals = np.tile(normal, (N, 1)).astype(np.float32)
    cosNrm = np.multiply(result["directionOut"], normal[None, :]).sum(1).astype(np.float32)
    trans = (cosNrm < 0.0) & ~absorbed
    refl = (cosNrm > 0.0) & ~absorbed
    ni = waterModel.refractive_index(lam) * np.ones(N)
    no = np.ones(N)

    # check we prevent self intersection
    cosPos = np.multiply(result["positionOut"], normal[None, :]).sum(-1)
    assert np.all(cosPos[trans] < 0.0)
    assert np.all(cosPos[refl] > 0.0)
    assert np.all(result["mediumIdxOut"][trans] == VACUUM_IDX)
    assert np.all(result["mediumIdxOut"][refl] == waterIdx)

    if not particle:
        c = result["contribIn"]
    if camera:
        eta = ni / no
        c = np.copy(result["contribIn"])
        c[trans] = (c * eta * eta)[trans]

    if microfacet_sampler is None:
        # no sampler: geometric checks only
        if flags == "R" or flags == "DR":
            assert trans.sum() == 0
        if flags == "T":
            assert refl.sum() == 0
        if flags == "TR":
            assert absorbed.sum() == 0
        return

    ref_normals = microfacet_sampler(normals, tangential_vector, direction, alpha, N).astype(np.float32)
    t = refract_arr(result["directionIn"], ref_normals, ni, no).astype(np.float32)
    r = reflect_arr(result["directionIn"], ref_normals).astype(np.float32)
    R = reflectance_arr(result["directionIn"], ref_normals, ni, no).astype(np.float32)
    cos_in = np.sum(result["directionIn"] * ref_normals, axis=1).astype(np.float32)
    cos_t = np.sum(t * normal[None, :], axis=1).astype(np.float32)
    cos_r = np.sum(r * normal[None, :], axis=1).astype(np.float32)
    if masking_function is not None:
        # VNDF sampling guarantees the incoming ray hits the microfacet from the front;
        # apply stochastic masking to the outgoing direction only.
        u_r = np.random.uniform(0.0, 1.0, N)
        u_t = np.random.uniform(0.0, 1.0, N)
        mask_r = (cos_r > 0) & (masking_function(np.abs(cos_r), alpha) > u_r)
        mask_t = (cos_t < 0) & (R < 1) & (masking_function(np.abs(cos_t), alpha) > u_t)
    else:
        mask_r = (cos_in < 0) & (cos_r > 0)
        mask_t = (cos_in < 0) & (cos_t < 0) & (R < 1)

    err = 0.02

    # Direction comparisons are only reliable when there are enough reference samples.
    enough_t = (mask_t.sum() >= 20000) and (trans.sum() >= 20000)
    enough_r = (mask_r.sum() >= 20000) and (refl.sum() >= 20000)

    # check correct sampling of micro-facets
    if (flags == "R" or flags == "DR") and not particle and enough_r:
        microfacet_normals_shader = result["directionOut"] - result["directionIn"]
        microfacet_normals_shader /= np.linalg.norm(microfacet_normals_shader, axis=1, keepdims=True)
        cos_test = np.clip(np.sum(normals[mask_r] * ref_normals[mask_r], axis=1), 0.0, 1.0)
        cos_shader = np.clip(np.sum(normals * microfacet_normals_shader, axis=1), 0.0, 1.0)
        assert np.mean(np.arccos(cos_test)) == pytest.approx(np.mean(np.arccos(cos_shader)), rel=err)

    R_mean_r = np.mean(R[mask_r])
    R_mean_t = np.mean(R[mask_t])

    if enough_t and (flags == "TR" or (flags == "T" and particle)):
        assert np.mean(result["directionOut"][trans], axis=0) == pytest.approx(
            np.average(t[mask_t], weights=(1 - R[mask_t]), axis=0), rel=err
        )
    if enough_t and (flags == "T") and not particle:
        assert np.mean(result["directionOut"][trans], axis=0) == pytest.approx(
            np.average(t[mask_t], axis=0), rel=err
        )
    if enough_r and (flags == "TR" or ((flags == "R" or flags == "DR") and particle)):
        assert np.mean(result["directionOut"][refl], axis=0) == pytest.approx(
            np.average(r[mask_r], weights=R[mask_r], axis=0), rel=err
        )
    if enough_r and (flags == "R" or flags == "DR") and not particle:
        assert np.mean(result["directionOut"][refl], axis=0) == pytest.approx(
            np.average(r[mask_r], axis=0), rel=err
        )

    if flags == "R" or flags == "DR":
        assert trans.sum() == 0
        if particle and enough_r:
            assert absorbed.sum() > 0
        elif enough_r:
            cO = result["contribOut"]
            assert np.mean(cO[~absorbed]) == pytest.approx(R_mean_r * np.mean(c[~absorbed]), rel=err)
    if flags == "T":
        assert refl.sum() == 0
        if particle and enough_t:
            absorbed.sum() > 0
        if not particle and enough_t:
            cO = result["contribOut"]
            assert np.mean(cO[~absorbed]) == pytest.approx((1 - R_mean_t) * np.mean(c[~absorbed]), rel=err)
        if not particle and not camera and enough_t:
            cH = result["contribHit"]
            assert np.mean(cH[~absorbed]) == pytest.approx((1 - R_mean_t) * np.mean(c[~absorbed]), rel=err)
    if flags == "TR":
        if enough_r:
            assert refl.sum() > 0
        if enough_t:
            assert trans.sum() > 0
        assert absorbed.sum() == 0
        if particle:
            np.all((result["hitSuccess"] == 0) == refl)
        else:
            assert np.allclose(result["contribOut"], c)


@pytest.mark.parametrize("particle,camera,flags,angle", _ROUGH_DIELECTRIC_PARAMS)
def test_DielectricBeckmannSurface(particle: bool, camera: bool, flags: str, angle: float):
    _test_rough_dielectric_surface(
        theia.surface.DielectricBeckmannSurface(),
        sample_microfacet_normals_beckmann,
        particle, camera, flags, angle,
    )


@pytest.mark.parametrize("particle,camera,flags,angle", _ROUGH_DIELECTRIC_PARAMS)
def test_DielectricTrowbridgeReitzSurface(particle: bool, camera: bool, flags: str, angle: float):
    _test_rough_dielectric_surface(
        theia.surface.DielectricTrowbridgeReitzSurface(),
        sample_microfacet_normals_trowbridge_reitz,
        particle, camera, flags, angle,
    )


@pytest.mark.parametrize("particle,camera,flags,angle", _ROUGH_DIELECTRIC_PARAMS)
def test_DielectricTrowbridgeReitzShadowedSurface(particle: bool, camera: bool, flags: str, angle: float):
    _test_rough_dielectric_surface(
        theia.surface.DielectricTrowbridgeReitzShadowedSurface(),
        sample_microfacet_normals_trowbridge_reitz_shadowed,
        particle, camera, flags, angle,
        masking_function=_masking_function_trowbridge_reitz,
    )
