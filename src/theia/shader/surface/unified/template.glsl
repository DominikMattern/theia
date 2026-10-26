#include "math.glsl"
#include "util/sample.glsl"
#include "surface/fresnel.glsl"
#include "surface/unified/lobes.glsl"

/*
Implementation of the Geant4 UNIFIED surface model, shared by its dielectric and
its metal variant (`SURFACE_UNIFIED_TRANSMISSION` selects the former). It differs
from the other rough surface models in two ways, both taken from Geant4:

 - The reflection is decomposed into a specular spike, specular lobe, diffuse
   lobe and backscattering, weighted by the material properties `prob_*`
   (see lobes.glsl).
 - A facet whose outgoing direction does not leave the micro surface does not
   restart the interaction. The walk continues with the direction that facet
   produced, which is Geant4's stand-in for multiple scattering on the micro
   structure. For the dielectric variant the walk also follows the ray across the
   interface.

Both make the model unreachable for NEE: the spike and backscatter components are
delta distributions no shadow ray can hit, and the walk has no closed-form
density.

The facets always come from the Gaussian slope-angle distribution.
*/

//The lobe decomposition has delta components no shadow ray can reach and the walk
//has no tractable density -> tell the tracer to take its specular path.
#define SURFACE_MODEL_SPECULAR

struct SurfaceProperties {
    vec3 dirOut;        ///< direction in which the walk left the micro surface
#ifdef SURFACE_UNIFIED_TRANSMISSION
    bool doReflect;     ///< true if the walk left on the side it came from
#else
    float reflectance;  ///< reflectance of the whole walk
    bool doReflect;     ///< sampled reflect-or-absorb decision (particles only)
#endif
    bool valid;         ///< false if the walk failed and the ray is absorbed
};

//tell tracer we want to do some prep work
#define SurfaceProperties SurfaceProperties

SurfaceProperties prepareSurface(
    const RAY ray,
    const SurfaceHit hit,
    uint idx, inout uint dim
) {
    //fetch optical constants of both sides
    float n_i = lookUpMediaTable1D(REFRACTIVE_INDEX, ray.mediumIdx, ray.wavelength, 1.0);
    float n_o = lookUpMediaTable1D(REFRACTIVE_INDEX, hit.otherMediumIdx, ray.wavelength, 1.0);
#ifndef SURFACE_UNIFIED_TRANSMISSION
    bool canReflect = (hit.flags & NO_REFLECT_BIT) == 0;
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
#endif

    SurfaceProperties props;
    props.dirOut = vec3(0.0);
    props.doReflect = false;
    props.valid = false;
#ifndef SURFACE_UNIFIED_TRANSMISSION
    props.reflectance = 0.0;

    //a metal that may not reflect absorbs everything
    if (!canReflect)
        return props;
#endif

    //Which reflection lobe to use is drawn at every reflection event for the
    //dielectric variant, but only once per walk for the metal (as in Geant4).
    //
    //The diffuse direction is pre-sampled once: every lobe but the specular one
    //leaves the micro surface immediately, so it is used at most once per walk.
#ifdef SURFACE_UNIFIED_TRANSMISSION
    vec3 diffuseDir = createLocalCOSY(hit.rayNrm) * sampleHemisphereCosine(random2D(idx, dim));
#else
    ReflectionLobes lobes = sampleReflectionLobes(hit, idx, dim);
    vec3 diffuseDir = vec3(0.0);
    if (lobes.doDiffuseLobe)
        diffuseDir = createLocalCOSY(hit.rayNrm) * sampleHemisphereCosine(random2D(idx, dim));
#endif

    //compute the facet-independent micro-facet parameters once
    MicrofacetParams mfParams = prepareMicrofacet(ray.direction, hit);

    //Product of the facet reflectances; only the metal loses energy. As in Geant4,
    //a REFLECTIVITY table applies once per hit, the Fresnel reflectance per facet.
#ifdef SURFACE_UNIFIED_TRANSMISSION
    float weight = 1.0;
#else
    float weight = hasReflectivity ? reflectivity : 1.0;
#endif
    //direction the ray currently travels in and the side of the interface it is on
    vec3 dir = ray.direction;
    bool farSide = false;
    //After 20 iterations, the next iteration uses the macroscopic normal, which always
    //sends the ray out of the micro surface, so the walk cannot run past it. Geant4
    //instead loops until it succeeds (with a warning after 100 boundary actions).
    for (uint i = 0; i <= 20; i++) {
        //macroscopic normal and refractive indices as seen from the current side
        vec3 nrm = farSide ? -hit.rayNrm : hit.rayNrm;
        float nIn = farSide ? n_o : n_i;
        float nOut = farSide ? n_i : n_o;

        //Sample a facet, mirrored to the current side. The last iteration is the
        //fall-back and uses the macroscopic normal instead.
        vec3 microfacetNormal = nrm;
        if (i < 20) {
            microfacetNormal = sampleMicrofacetNormal(mfParams, dir, hit, idx, dim);
            if (farSide) microfacetNormal = -microfacetNormal;
            //Geant4's GetFacetNormal() redraws until the facet faces the photon.
            //That leaves the direction untouched, so unlike the walk below this
            //really is a plain facet rejection.
            if (dot(dir, microfacetNormal) >= 0.0) continue;
        }
        float cos_i = -dot(dir, microfacetNormal);

#ifdef SURFACE_UNIFIED_TRANSMISSION
        //returns 1.0 in total internal reflection
        float F = fresnelReflectance(cos_i, nIn, nOut);
        bool doReflect = random(idx, dim) < F;
        ReflectionLobes lobes = sampleReflectionLobes(hit, idx, dim);
#else
        //metals never transmit; the non-reflected fraction is absorbed
        bool doReflect = true;
        if (!hasReflectivity)
            weight *= fresnelConductor(cos_i, nIn, nOut, k);
#endif

        if (doReflect) {
            //the diffuse lobe scatters around the macroscopic normal of the
            //current side, so the pre-sampled direction has to be mirrored too
            dir = lobeReflectedDir(
                lobes, dir, nrm, microfacetNormal, farSide ? -diffuseDir : diffuseDir);
        }
        else {
            vec3 dirRefracted = refract(dir, microfacetNormal, nIn / nOut);
            //A zero direction marks total internal reflection, which the Fresnel
            //coin above should have excluded. Due to finite numerical precision
            //that edge is a bit fuzzy -> mark as absorbed.
            if (dirRefracted == vec3(0.0))
                return props;
            //the ray went through the facet and is now on the other side
            dir = dirRefracted;
            farSide = !farSide;
        }

        //the walk ends as soon as the direction leaves the micro surface
        if (dot(dir, farSide ? -hit.rayNrm : hit.rayNrm) > 0.0) {
            props.valid = true;
            break;
        }
    }

    props.dirOut = dir;
#ifdef SURFACE_UNIFIED_TRANSMISSION
    //the ray left on the far side exactly if it crossed the interface an odd
    //number of times
    props.doReflect = !farSide;
#else
    props.reflectance = props.valid ? weight : 0.0;
    //Only used by particle rays, which cannot carry a weight. Drawn once for the
    //whole walk rather than per bounce - same statistics, fewer draws.
    props.doReflect = random(idx, dim) < props.reflectance;
#endif
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
#ifdef SURFACE_UNIFIED_TRANSMISSION

    /*
    A walk that continues as a reflection never reached the detector, everything
    else did: a transmission, or an outcome the flags absorb.
    */
    bool canReflect = (hit.flags & NO_REFLECT_BIT) == 0;
    if (props.valid && props.doReflect && canReflect)
        return false;

    //if requested, transmit ray before detecting
    //(this does not change ray contribution)
    bool transmitHit = (hit.flags & MATERIAL_TRANSMIT_HIT_BIT) != 0;
    if (transmitHit && props.valid && !props.doReflect) {
        transmitRay(ray, hit, props.dirOut);
    }
    item = createHit(
        ray,
        hit.objPos,
        hit.objNrm,
        objectId,
        hit.worldToObj
    );
    return true;

#elif defined(RAY_PARTICLE)

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
    if (!props.valid)
        return RESULT_CODE_RAY_ABSORBED;

    ResultCode result;
#ifdef SURFACE_UNIFIED_TRANSMISSION

    //fetch material flags
    bool isDetector = (hit.flags & MATERIAL_DETECTOR_BIT) != 0;
    bool canReflect = (hit.flags & NO_REFLECT_BIT) == 0;
    bool canTransmit = (hit.flags & NO_TRANSMIT_BIT) == 0;
    //detector implies no transmit
    canTransmit = !isDetector && canTransmit;

    if (props.doReflect) {
        if (!canReflect)
            return RESULT_CODE_RAY_ABSORBED;
        result = reflectRay(ray, hit, props.dirOut);
    }
    else {
        if (!canTransmit)
            return RESULT_CODE_RAY_ABSORBED;
        result = transmitRay(ray, hit, props.dirOut);
    }

#else

    #ifdef RAY_PARTICLE
    //importance sampled whether to reflect or get absorbed
    if (!props.doReflect)
        return RESULT_CODE_RAY_ABSORBED;
    #else
    if (props.reflectance <= 0.0)
        return RESULT_CODE_RAY_ABSORBED;
    //only reflection allowed -> deterministically reflect, attenuate by reflectance
    ray.lin_contrib *= props.reflectance;
    #endif
    result = reflectRay(ray, hit, props.dirOut);

#endif

    //success
    return result >= 0 ? RESULT_CODE_RAY_HIT : result;
}
