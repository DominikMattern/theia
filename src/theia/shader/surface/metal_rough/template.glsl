#include "math.glsl"
#include "util/sample.glsl"
#include "surface/fresnel.glsl"
#include "surface/lobes.glsl"

/*
Shared implementation of all rough metal surface models. The only model-specific
parts are `sample_microfacet_normal` and `check_microfacet`, which are provided
by one of the files in `microfacet_models/`.

Metals never transmit; the non-reflected fraction is absorbed. The reflected
component is decomposed into a specular spike, specular lobe, diffuse lobe and
backscattering (see `surface/lobes.glsl`).

The reflectance is either taken from the optional material property
`reflectivity` or, if not given, computed from the conductor Fresnel equations
using the complex refractive index of the metal medium (real part =
`refractive_index`, imaginary part = `extinction_coefficient` of the other side).
*/

struct SurfaceProperties {
    float reflectance;
    vec3 dirReflected;
    bool doReflect;
};

//tell tracer we want to do some prep work
#define SurfaceProperties SurfaceProperties

SurfaceProperties prepareSurface(
    const RAY ray,
    const SurfaceHit hit,
    uint idx, inout uint dim
) {
    //fetch refractive indices (incident medium n_i, metal medium n_o + extinction k)
    float n_i = lookUpMediaTable1D(REFRACTIVE_INDEX, ray.mediumIdx, ray.wavelength, 1.0);
    float n_o = lookUpMediaTable1D(REFRACTIVE_INDEX, hit.otherMediumIdx, ray.wavelength, 1.0);
    float k = 0.0;
    #ifdef MEDIA_SLOT_EXTINCTION_COEFFICIENT
        k = lookUpMediaTable1D(EXTINCTION_COEFFICIENT, hit.otherMediumIdx, ray.wavelength, 0.0);
    #endif

    //metals only reflect
    bool canReflect = (hit.flags & NO_REFLECT_BIT) == 0;

    //sample which reflection lobe to use (specular spike/lobe, diffuse, backscatter)
    ReflectionLobes lobes = sampleReflectionLobes(hit, idx, dim);

    //fetch optional reflectivity. If not provided, use the conductor Fresnel reflectance.
    bool hasReflectivity = false;
    float reflectivity = 0.0;
    #ifdef MATERIAL_SLOT_REFLECTIVITY
        reflectivity = lookUpMaterialTable1D(REFLECTIVITY, hit.materialIdx, ray.wavelength, -1.0);
        //lookUp returns -1.0 if this material has no reflectivity table
        hasReflectivity = (reflectivity >= 0.0);
    #endif

    //pre-sample the diffuse hemisphere direction once (only used by the diffuse lobe)
    vec3 diffuseDir = vec3(0.0);
    if (lobes.doDiffuseLobe)
        diffuseDir = createLocalCOSY(hit.rayNrm) * sampleHemisphereCosine(random2D(idx, dim));

    //we only need a micro-facet normal if the specular lobe direction is needed
    //or if we have to compute the Fresnel reflectance
    bool needMicrofacet = lobes.doSpecularLobe || !hasReflectivity;

    float r = hasReflectivity ? reflectivity : 0.0;
    vec3 dirReflected = vec3(0.0);
    bool doReflect = false;

    if (needMicrofacet) {
        //sample micro-facets until a valid one is found (in most cases the first is valid)
        for (int i = 0; i < 8; i++) {
            vec3 microfacetNormal = sample_microfacet_normal(ray.direction, hit, idx, dim);

            //conductor Fresnel reflectance at the micro-facet (unless a reflectivity is given)
            float cos_i = abs(dot(ray.direction, microfacetNormal));
            if (!hasReflectivity)
                r = fresnelConductor(cos_i, n_i, n_o, k);

            doReflect = random(idx, dim) < r;

            bool cond1 = true;
            bool cond2 = true;

            if (canReflect) {
                //actual outgoing direction of the sampled reflection lobe
                dirReflected = lobeReflectedDir(lobes, ray.direction, hit.rayNrm, microfacetNormal, diffuseDir);
                //check that reflected ray returns to the original medium
                cond1 = (dot(dirReflected, hit.rayNrm) > 0);
                //model-dependent validity check
                vec3 dirReflectedMicrofacet = reflect(ray.direction, microfacetNormal);
                cond2 = check_microfacet(dirReflectedMicrofacet, microfacetNormal, hit, ray.direction, idx, dim);
            }

            if (cond1 && cond2)
                break;

            //after 8 invalid micro-facets have been sampled, an unrotated facet is chosen to continue
            if (i == 7) {
                float cos_i2 = abs(dot(ray.direction, hit.rayNrm));
                if (!hasReflectivity)
                    r = fresnelConductor(cos_i2, n_i, n_o, k);
                dirReflected = lobeReflectedDir(lobes, ray.direction, hit.rayNrm, hit.rayNrm, diffuseDir);
                doReflect = random(idx, dim) < r;
            }
        }
    }
    else {
        //reflectivity given and no micro-facet needed (no specular lobe)
        doReflect = random(idx, dim) < r;
        dirReflected = lobeReflectedDir(lobes, ray.direction, hit.rayNrm, hit.rayNrm, diffuseDir);
    }

    //return properties
    return SurfaceProperties(r, dirReflected, doReflect);
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
