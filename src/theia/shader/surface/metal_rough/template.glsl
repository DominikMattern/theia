#include "math.glsl"
#include "util/sample.glsl"
#include "surface/fresnel.glsl"

/*
Shared implementation of all rough metallic surface models. The only
model-specific parts are `prepareMicrofacet`, `sampleMicrofacetNormal` and
`microfacetAcceptProb`, which are provided by one of the files in
`microfacet_models/` (see api.glsl there).

Metals never transmit; the non-reflected fraction is absorbed. The reflectance is
either taken from the optional material property `reflectivity` or, if not given,
computed from the conductor Fresnel equations using the complex refractive index
of the metal medium (real part = `refractive_index`, imaginary part =
`extinction_coefficient` of the other side).

The Geant4 UNIFIED model, which splits the reflection into several lobes and
walks across the micro structure instead of rejecting facets, lives in
surface/unified/ instead.
*/

#define SURFACE_MODEL_SPECULAR

struct SurfaceProperties {
    float weight;           ///< reflectance at the accepted facet, 0 if absorbed
    vec3 dirReflected;
    bool doReflect;         ///< sampled reflect-or-absorb decision (particles only)
};

//tell tracer we want to do some prep work
#define SurfaceProperties SurfaceProperties

SurfaceProperties prepareSurface(
    const RAY ray,
    const SurfaceHit hit,
    uint idx, inout uint dim
) {

    SurfaceProperties props;
    props.weight = 0.0;
    props.dirReflected = vec3(0.0);
    props.doReflect = false;

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

    /*
    We use the following rejection sampling: sample microfacet -> check if the
    reflected direction is valid -> either use that direction or resample

    The underlying assumption is that invalid directions correspond to multi-
    scattering, and that the distribution of the twice scattered rays is identical
    to single scattered rays (the following paper argued that this is roughly 
    correct: https://api.semanticscholar.org/CorpusID:221737278).
    */

    //compute the facet-independent micro-facet parameters once
    MicrofacetParams mfParams = prepareMicrofacet(ray.direction, hit);

    //Sample micro-facets until one is accepted; in most cases the first one is.
    //The extra iteration is the fall-back after 20 failed attempts: an unrotated
    //facet (the surface normal), accepted as long as it reflects at all.
    bool valid = false;
    vec3 acceptedNormal = hit.rayNrm;
    for (uint i = 0; i <= 20; i++) {
        bool last = (i == 20);
        vec3 microfacetNormal = hit.rayNrm;
        if (!last)
            microfacetNormal = sampleMicrofacetNormal(mfParams, ray.direction, hit, idx, dim);

        vec3 dir = reflect(ray.direction, microfacetNormal);
        //acceptance probability
        float a = dot(dir, hit.rayNrm) > 0.0
            ? microfacetAcceptProb(mfParams, dir, microfacetNormal, hit, ray.direction)
            : 0.0;
        float u = random(idx, dim);

        if (last ? (a > 0.0) : (u < a)) {
            valid = true;
            acceptedNormal = microfacetNormal;
            props.dirReflected = dir;
            break;
        }
    }

    //can only happen if the unrotated facet does not reflect either
    if (!valid)
        return props;

    //reflectance at the accepted facet
    float cos_i = abs(dot(ray.direction, acceptedNormal));
    props.weight = hasReflectivity ? reflectivity : fresnelConductor(cos_i, n_i, n_o, k);
    //Only used by particle rays, which cannot carry a weight.
    props.doReflect = random(idx, dim) < props.weight;

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

    //We have a local copy of the ray. Attenuate before detecting: what is not
    //reflected back is what enters the detector. sampleSurfaceInteraction()
    //passes props.weight on to the reflected ray, so the complement is exactly
    //what the surface does not send back.
    ray.lin_contrib *= (1.0 - props.weight);
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

    if (!canReflect || props.weight <= 0.0)
        return RESULT_CODE_RAY_ABSORBED;

    #ifdef RAY_PARTICLE
    //importance sampled whether to reflect or get absorbed
    if (!props.doReflect)
        return RESULT_CODE_RAY_ABSORBED;
    #else
    //deterministically reflect, attenuate by reflectance
    ray.lin_contrib *= props.weight;
    #endif
    ResultCode result = reflectRay(ray, hit, props.dirReflected);

    //success
    return result >= 0 ? RESULT_CODE_RAY_HIT : result;
}
