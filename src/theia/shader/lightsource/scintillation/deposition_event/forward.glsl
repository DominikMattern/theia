#ifndef _INCLUDE_LIGHTSOURCE_SCINTILLATION_DEPOSITIONEVENT_FORWARD
#define _INCLUDE_LIGHTSOURCE_SCINTILLATION_DEPOSITIONEVENT_FORWARD

#include "math.glsl"
#include "util/sample.glsl"
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

    //sample direction
    vec3 direction = sampleUnitSphere(random2D(idx, dim));

    //sample scintillation time delay from exponential decay
    float v = random(idx, dim);
    float total_time = time - log(1.0 - v) * lightParams.timeConstant[batchIdx];

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
