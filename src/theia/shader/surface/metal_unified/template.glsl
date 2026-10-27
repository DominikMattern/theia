#include "math.glsl"
#include "util/sample.glsl"
#include "surface/fresnel.glsl"
#include "surface/unified/lobes.glsl"

/*
Implementation of the Geant4 UNIFIED surface model for a rough dielectric-metal
interface. It differs from the other rough surface models in two ways, both taken
from Geant4:

 - The reflection is decomposed into a specular spike, specular lobe, diffuse
   lobe and backscattering, weighted by the material properties `prob_*`
   (see surface/unified/lobes.glsl).
 - A facet whose outgoing direction does not leave the micro surface does not
   restart the interaction. The walk continues with the direction that facet
   produced, which is Geant4's stand-in for multiple scattering on the micro
   structure.

Metals never transmit; the non-reflected fraction is absorbed. The reflectance is
either taken from the optional material property `reflectivity`, applied once per
hit, or, if not given, computed from the conductor Fresnel equations at every
facet the walk reflects off, using the complex refractive index of the metal
medium (real part = `refractive_index`, imaginary part = `extinction_coefficient`
of the other side).

The facets always come from the Gaussian slope-angle distribution.
*/

#define SURFACE_MODEL_SPECULAR

struct SurfaceProperties {
    float weight;           ///< reflectance of the whole walk, 0 if absorbed
    vec3 dirReflected;      ///< direction in which the walk left the micro surface
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

    //Unlike the dielectric, which reflection lobe to use is drawn only once per
    //walk (as in Geant4).
    //
    //The diffuse direction is pre-sampled once: every lobe but the specular one
    //leaves the micro surface immediately, so it is used at most once per walk.
    ReflectionLobes lobes = sampleReflectionLobes(hit, idx, dim);
    vec3 diffuseDir = vec3(0.0);
    if (lobes.doDiffuseLobe)
        diffuseDir = createLocalCOSY(hit.rayNrm) * sampleHemisphereCosine(random2D(idx, dim));

    //compute the facet-independent micro-facet parameters once
    MicrofacetParams mfParams = prepareMicrofacet(ray.direction, hit);

    //Product of the facet reflectances. As in Geant4, a REFLECTIVITY table
    //applies once per hit, the Fresnel reflectance per facet.
    float weight = hasReflectivity ? reflectivity : 1.0;
    //direction the ray currently travels in
    vec3 dir = ray.direction;
    bool valid = false;
    //After 20 iterations, the next iteration uses the macroscopic normal, which always
    //sends the ray out of the micro surface, so the walk cannot run past it. Geant4
    //instead loops until it succeeds (with a warning after 100 boundary actions).
    for (uint i = 0; i <= 20; i++) {
        //Sample a facet. The last iteration is the fall-back and uses the
        //macroscopic normal instead.
        vec3 microfacetNormal = hit.rayNrm;
        if (i < 20) {
            microfacetNormal = sampleMicrofacetNormal(mfParams, dir, hit, idx, dim);
            //Geant4's GetFacetNormal() redraws until the facet faces the photon.
            //That leaves the direction untouched, so unlike the walk below this
            //really is a plain facet rejection.
            if (dot(dir, microfacetNormal) >= 0.0) continue;
        }

        //metals never transmit; the non-reflected fraction is absorbed
        float cos_i = -dot(dir, microfacetNormal);
        if (!hasReflectivity)
            weight *= fresnelConductor(cos_i, n_i, n_o, k);

        dir = lobeReflectedDir(lobes, dir, hit.rayNrm, microfacetNormal, diffuseDir);

        //the walk ends as soon as the direction leaves the micro surface
        if (dot(dir, hit.rayNrm) > 0.0) {
            valid = true;
            break;
        }
    }

    //can only happen if the walk did not leave the micro surface
    if (!valid)
        return props;

    props.weight = weight;
    props.dirReflected = dir;
    //Only used by particle rays, which cannot carry a weight. Drawn once for the
    //whole walk rather than per bounce - same statistics, fewer draws.
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
