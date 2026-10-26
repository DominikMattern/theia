#ifndef _INCLUDE_LIGHTSOURCE_SCINTILLATION_DEPOSITIONEVENT_FORWARD
#define _INCLUDE_LIGHTSOURCE_SCINTILLATION_DEPOSITIONEVENT_FORWARD

#include "math.glsl"
#include "util/sample.glsl"
//SPEED_OF_LIGHT and lookUpMediaTable1D for the Cherenkov cone; guarded, so a
//second include costs nothing
#include "material.glsl"
#include "lightsource/scintillation/deposition_event/common.glsl"

ForwardRay sampleLight(uint idx, inout uint dim) {

    //sample wavelength using wavelength source
    #ifdef LIGHT_SOURCE_EMIT_PARTICLE
    float wavelength = sampleWavelength(idx, dim);
    #else
    float contrib;
    float wavelength = sampleWavelength(contrib, idx, dim);
    #endif

    //select batch element

    #if defined(LIGHT_SOURCE_EMIT_PARTICLE) || defined(LIGHT_SOURCE_DETERMINISTIC_RAYS)

    //Safeguard: paths beyond the ones the elements own would be attributed to
    //the last element, as the batch size set on the host is not checked against
    //the counts. Reject them instead (zero direction).
    if (idx >= lightParams.cumulativeCount[EVENTS_PER_BATCH - 1]) {
        return createForwardRay(
            vec3(0.0),
            vec3(0.0),
            wavelength,
            lightParams.mediumIdx[EVENTS_PER_BATCH - 1],
            0.0
            #ifndef LIGHT_SOURCE_EMIT_PARTICLE
            , 0.0
            #endif
        );
    }

    int batchIdx = findBatchElement(idx);

    #elif EVENTS_PER_BATCH > 1

    //importance sample a batch element
    int x = int(floor(random(idx, dim) * float(lightParams.cumulativePhotonNumber[EVENTS_PER_BATCH-1])));

    //binary search
    int lo = 0;
    int hi = EVENTS_PER_BATCH - 1;

    while (lo < hi)
    {
        int mid = (lo + hi) >> 1;

        if (x < int(lightParams.cumulativePhotonNumber[mid]))
            hi = mid;
        else
            lo = mid + 1;
    }

    int batchIdx = lo;

    #else

    int batchIdx = 0;

    #endif

    //sample position and time along the track segment
    float u = random(idx,dim);
    vec3 position = mix(lightParams.startPosition[batchIdx], lightParams.endPosition[batchIdx], u);
    float time = lightParams.startTime[batchIdx]
                + length(position - lightParams.startPosition[batchIdx])
                / max((lightParams.startVelocity[batchIdx] + u * (lightParams.endVelocity[batchIdx] - lightParams.startVelocity[batchIdx]) / 2), 1e-5);

    //sample direction and emission time
    vec3 direction;
    float total_time;
    #ifdef LIGHT_SOURCE_CHERENKOV
    if (lightParams.mode[batchIdx] != 0u) {
        //Cherenkov: a cone about the segment direction with cos(theta) = 1/(beta*n)
        float beta = 0.5 * (lightParams.startVelocity[batchIdx]
                          + lightParams.endVelocity[batchIdx]) / SPEED_OF_LIGHT;
        float n = lookUpMediaTable1D(
            REFRACTIVE_INDEX, lightParams.mediumIdx[batchIdx], wavelength, 1.0);
        float cos_theta = 1.0 / max(beta * n, 1.0);
        float phi = TWO_PI * random(idx, dim);
        vec3 trackDir = normalize(lightParams.endPosition[batchIdx]
                                - lightParams.startPosition[batchIdx]);
        direction = createLocalCOSY(trackDir) * sphericalToCartessian(phi, cos_theta);
        //prompt: there is no decay to wait for
        total_time = time;
    }
    else
    #endif
    {
        direction = sampleUnitSphere(random2D(idx, dim));

        //sample scintillation time delay from exponential decay
        float v = random(idx, dim);
        total_time = time - log(1.0 - v) * lightParams.timeConstant[batchIdx];
    }

    //calculate contribution
    #ifndef LIGHT_SOURCE_EMIT_PARTICLE
    #ifdef LIGHT_SOURCE_DETERMINISTIC_RAYS
    contrib *= lightParams.photonsPerRay[batchIdx]
             * float(lightParams.cumulativeCount[EVENTS_PER_BATCH-1]);
    #else
    contrib *= lightParams.cumulativePhotonNumber[EVENTS_PER_BATCH-1];
    #endif
    #ifdef SCINTILLATION_USE_ENERGY
    // To get from photon count to energy in eV, multiply by
    //    h * c             h * c
    // ------------  with  -------- = 1.239841984e-6
    //  e * lambda            e
    contrib *= 1.239841983e-6 / wavelength;
    #endif
    #endif

    //create forward ray
    return createForwardRay(
        position,
        direction,
        wavelength,
        lightParams.mediumIdx[batchIdx],
        total_time
        #ifndef LIGHT_SOURCE_EMIT_PARTICLE
        , contrib
        #endif
    );
}

#endif
