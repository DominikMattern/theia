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

    #ifdef LIGHT_SOURCE_EMIT_PARTICLE
    //for light particles, the event is chosen based on the thread idx to ensure
    //that each event gets the correct number of photons

    if (idx >= int(lightParams.cumulativePhotonNumber[BATCH_SIZE-1])){
        contrib = 0.0;
    }

    //binary search
    int lo = 0;
    int hi = BATCH_SIZE - 1;

    while (lo < hi)
    {
        int mid = (lo + hi) >> 1;

        if (idx < int(lightParams.cumulativePhotonNumber[mid]))
            hi = mid;
        else
            lo = mid + 1;
    }

    int batchIdx = lo;

    #elif BATCH_SIZE > 1

    //importance sample a batch element
    int x = int(floor(random(idx, dim) * float(lightParams.cumulativePhotonNumber[BATCH_SIZE-1])));

    //binary search
    int lo = 0;
    int hi = BATCH_SIZE - 1;

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
    float total_time = time - log(1.0 - v) * lightParams.timeConstant;

    //calculate contribution
    contrib = dw_dA(position, observer) * INV_4PI;

    #ifdef SCINTILLATION_USE_ENERGY
    // To get from photon count to energy in eV, multiply by
    //    h * c             h * c
    // ------------  with  -------- = 1.239841984e-6
    //  e * lambda            e
    contrib *= lightParams.cumulativePhotonNumber[BATCH_SIZE-1] * 1.239841983e-6 / wavelength;
    #else
    contrib *= lightParams.cumulativePhotonNumber[BATCH_SIZE-1];
    #endif

    //assemble ray
    return createForwardRay(
        position,
        direction,
        wavelength,
        lightParams.mediumIdx, //do not use function parameter here. We might be in a different medium!
        total_time,
        contrib
    );
}

#endif
