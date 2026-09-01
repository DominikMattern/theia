#ifndef _INCLUDE_LIGHTSOURCE_SCINTILLATION_DEPOSITIONEVENT_BACKWARD
#define _INCLUDE_LIGHTSOURCE_SCINTILLATION_DEPOSITIONEVENT_BACKWARD

#include "util/jacobian.glsl"
#include "math.glsl"
#include "lightsource/scintillation/deposition_event/common.glsl"

ForwardRay sampleLight(
    vec3 observer, vec3 normal,
    float wavelength,
    uint mediumIdx,
    uint idx, inout uint dim
) {

    float contrib = 1.0;

    ///select batch element

    #if defined(LIGHT_SOURCE_EMIT_PARTICLE) || defined(LIGHT_SOURCE_DETERMINISTIC_RAYS)

    if (idx >= lightParams.cumulativeCount[EVENTS_PER_BATCH-1]){
        contrib = 0.0;
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

    //get direction
    vec3 direction = normalize(observer - position);

    //sample startTime from exponential decay
    float v = random(idx, dim);
    float total_time = time - log(1.0 - v) * lightParams.timeConstant[batchIdx];

    //calculate contribution
    //multiply, do not assign: the out of range guard above zeroed contrib, and
    //assigning here would silently revive paths beyond the last element
    contrib *= dw_dA(position, observer) * INV_4PI;

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

    //assemble ray
    return createForwardRay(
        position,
        direction,
        wavelength,
        lightParams.mediumIdx[batchIdx],
        total_time,
        contrib
    );
}

#endif
