#include "math.glsl"
#include "util/sample.glsl"
#include "surface/fresnel.glsl"

/*
Shared implementation of all rough metal surface models. The only model-specific
parts are `prepareMicrofacet`, `sampleMicrofacetNormal`, `microfacetAcceptProb`
and, for NEE/MIS, `evalMicrofacetDensity`, which are provided by one of the files
in `microfacet_models/` (see api.glsl there).

Metals never transmit; the non-reflected fraction is absorbed. The surface is a
pure specular lobe on micro-facets and supports NEE/MIS. The Geant4 UNIFIED
model, which splits the reflection into several lobes and walks across the micro
structure instead of rejecting facets, lives in surface/unified/ instead.

The reflectance is either taken from the optional material property
`reflectivity` or, if not given, computed from the conductor Fresnel equations
using the complex refractive index of the metal medium (real part =
`refractive_index`, imaginary part = `extinction_coefficient` of the other side).
*/

//SURFACE_MODEL_SPECULAR and ROUGH_SURFACE_NEE are resolved in
//surface/rough_nee_config.glsl, which theia.surface prepends ahead of the
//micro-facet model so the model can key off them as well.

struct SurfaceProperties {
    float reflectance;
    vec3 dirReflected;
    bool doReflect;
    //Optical constants of the two sides, cached because the NEE functions need
    //them several times per surface hit and each lookup is a chain of dependent
    //memory accesses. They do not depend on the sampled facet. A negative
    //`reflectivity` means the material has no such table and the conductor
    //Fresnel equations apply.
    float n_i;
    float n_o;
    float k;
    float reflectivity;
};

//tell tracer we want to do some prep work
#define SurfaceProperties SurfaceProperties

float roughFresnel(float cos_i, const SurfaceProperties props) {
    //optional wavelength dependent reflectivity overrides the Fresnel equations
    if (props.reflectivity >= 0.0)
        return props.reflectivity;
    return fresnelConductor(cos_i, props.n_i, props.n_o, props.k);
}

/**
 * Outgoing direction of the reflection channel at the given facet and the
 * probability that the facet loop accepts it: the direction has to stay in the
 * upper macroscopic hemisphere and the model specific check has to pass. Metals
 * have no second channel, so this is the whole acceptance.
 */
float roughChannel(
    const MicrofacetParams mfParams,
    vec3 rayDir,
    const SurfaceHit hit,
    vec3 microfacetNormal,
    out vec3 dir
) {
    dir = reflect(rayDir, microfacetNormal);
    return dot(dir, hit.rayNrm) > 0.0
        ? microfacetAcceptProb(mfParams, dir, microfacetNormal, hit, rayDir)
        : 0.0;
}

//Probability that the facet loop accepts this facet. Consumed by
//surface/microfacet/normalization.glsl, which tabulates exactly this quantity -
//so the table cannot drift away from the loop. `eta` is unused: the
//reflect-or-absorb decision does not change the direction.
float roughFacetAcceptance(
    const MicrofacetParams mfParams, vec3 rayDir, const SurfaceHit hit,
    float eta, vec3 microfacetNormal
) {
    vec3 dir;
    return roughChannel(mfParams, rayDir, hit, microfacetNormal, dir);
}

#ifdef ROUGH_SURFACE_NEE
//Acceptance rate of the facet loop below, tabulated on the host over the
//incidence cosine (see theia.microfacet). Independent of the reflectance: the
//reflect-or-absorb decision does not change the direction and therefore does
//not enter the acceptance.
float roughNormalization(
    const RAY ray, const SurfaceHit hit, const SurfaceProperties props
) {
    float mu_i = abs(dot(ray.direction, hit.rayNrm));
    return lookUpMaterialTable1D(MICROFACET_Z_REFL, hit.materialIdx, mu_i, 0.0);
}
#endif

SurfaceProperties prepareSurface(
    const RAY ray,
    const SurfaceHit hit,
    uint idx, inout uint dim
) {
    //metals only reflect
    bool canReflect = (hit.flags & NO_REFLECT_BIT) == 0;

    //fetch refractive indices (incident medium n_i, metal medium n_o + extinction k)
    float n_i = lookUpMediaTable1D(REFRACTIVE_INDEX, ray.mediumIdx, ray.wavelength, 1.0);
    float n_o = lookUpMediaTable1D(REFRACTIVE_INDEX, hit.otherMediumIdx, ray.wavelength, 1.0);
    float k = 0.0;
    #ifdef MEDIA_SLOT_EXTINCTION_COEFFICIENT
        k = lookUpMediaTable1D(EXTINCTION_COEFFICIENT, hit.otherMediumIdx, ray.wavelength, 0.0);
    #endif

    //fetch optional reflectivity. If not provided, use the conductor Fresnel reflectance.
    bool hasReflectivity = false;
    float reflectivity = 0.0;
    #ifdef MATERIAL_SLOT_REFLECTIVITY
        reflectivity = lookUpMaterialTable1D(REFLECTIVITY, hit.materialIdx, ray.wavelength, -1.0);
        //lookUp returns -1.0 if this material has no reflectivity table
        hasReflectivity = (reflectivity >= 0.0);
    #endif

    SurfaceProperties props;
    props.reflectance = hasReflectivity ? reflectivity : 0.0;
    props.dirReflected = vec3(0.0);
    props.doReflect = false;
    props.n_i = n_i;
    props.n_o = n_o;
    props.k = k;
    props.reflectivity = hasReflectivity ? reflectivity : -1.0;

    //a metal that may not reflect absorbs everything
    if (!canReflect)
        return props;

    //compute the facet-independent micro-facet parameters once (roughness,
    //tangent basis, ...) and reuse them across all retries
    MicrofacetParams mfParams = prepareMicrofacet(ray.direction, hit);

    //Sample micro-facets until one is accepted; in most cases the first one is.
    //The extra iteration is the fall-back after 20 failed attempts: an unrotated
    //facet (the surface normal) whose acceptance is forced.
    for (int i = 0; i <= 20; i++) {
        bool last = (i == 20);
        vec3 microfacetNormal = hit.rayNrm;
        if (!last)
            microfacetNormal = sampleMicrofacetNormal(mfParams, ray.direction, hit, idx, dim);

        //reflectance at this facet
        float cos_i = abs(dot(ray.direction, microfacetNormal));
        if (!hasReflectivity)
            props.reflectance = fresnelConductor(cos_i, n_i, n_o, k);

        //decide whether this facet is accepted
        float a = roughChannel(
            mfParams, ray.direction, hit, microfacetNormal, props.dirReflected);
        float u = random(idx, dim);
        if (last ? (a > 0.0) : (u < a))
            break;
    }

    //Only used by particle rays, which cannot carry a weight. Drawn once for the
    //accepted facet rather than per attempt - same statistics, fewer draws.
    props.doReflect = random(idx, dim) < props.reflectance;

    return props;
}

bool processSurfaceTargetHit(
    RAY ray,
    const SurfaceHit hit,
    const SurfaceProperties props,
    int objectId,
    out HitItem item,
    uint idx, inout uint dim
) {
    #ifdef RAY_PARTICLE

    //we can only detect whole particles -> ignore if we sampled reflection earlier
    if (!props.doReflect) {
        item = createHit(
            ray,
            hit.objPos,
            hit.objNrm,
            objectId,
            hit.worldToObj
        );
    }
    return !props.doReflect;

    #else

    //we have a local copy of the ray. Attenuate by reflectance before detecting
    ray.lin_contrib *= (1.0 - props.reflectance);
    item = createHit(
        ray,
        hit.objPos,
        hit.objNrm,
        objectId,
        hit.worldToObj
    );
    return true;

    #endif
}

ResultCode sampleSurfaceInteraction(
    inout RAY ray,
    const SurfaceHit hit,
    const SurfaceProperties props,
    uint idx, inout uint dim
) {
    //metals only reflect
    bool canReflect = (hit.flags & NO_REFLECT_BIT) == 0;

    #ifdef RAY_PARTICLE

    //importance sample whether to reflect or get absorbed
    ResultCode result;
    if (props.doReflect && canReflect) {
        result = reflectRay(ray, hit, props.dirReflected);
    }
    else {
        //sampled decision is forbidden / absorbed -> abort tracing
        return RESULT_CODE_RAY_ABSORBED;
    }
    //success
    return result >= 0 ? RESULT_CODE_RAY_HIT : result;

    #else

    ResultCode result;
    if (canReflect && props.reflectance > 0.0) {
        //only reflection allowed -> deterministically reflect, attenuate by reflectance
        ray.lin_contrib *= props.reflectance;
        result = reflectRay(ray, hit, props.dirReflected);
    }
    else {
        return RESULT_CODE_RAY_ABSORBED;
    }
    //success
    return result >= 0 ? RESULT_CODE_RAY_HIT : result;

    #endif
}

#ifdef ROUGH_SURFACE_NEE
#include "surface/rough_nee.glsl"
#endif
