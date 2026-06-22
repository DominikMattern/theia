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

    //select batch element
    //energyContrib is the energy factor of the contribution, i.e. the deposited
    //energy of the selected segment divided by its selection probability
    #if defined(EVENT_IS) && BATCH_SIZE > 1
    //importance sampling: P(i) proportional to the deposited energy
    float totalEnergy = 0.0;
    for (int i = 0; i < BATCH_SIZE; ++i)
        totalEnergy += lightParams.energyDeposition[i];
    float r = random(idx, dim) * totalEnergy;
    int batchIdx = 0;
    float cumEnergy = lightParams.energyDeposition[0];
    while (batchIdx < BATCH_SIZE - 1 && r > cumEnergy) {
        batchIdx++;
        cumEnergy += lightParams.energyDeposition[batchIdx];
    }
    float energyContrib = totalEnergy;
    #elif BATCH_SIZE > 1
    //uniform sampling: P(i) = 1 / BATCH_SIZE
    int batchIdx = int(random(idx, dim) * float(BATCH_SIZE));
    float energyContrib = lightParams.energyDeposition[batchIdx] * float(BATCH_SIZE);
    #else
    int batchIdx = 0;
    float energyContrib = lightParams.energyDeposition[batchIdx];
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
    float contrib = dw_dA(position, observer) * INV_4PI;

    #ifdef SCINTILLATION_USE_ENERGY
    // To get from photon count to energy in eV, multiply by
    //    h * c             h * c
    // ------------  with  -------- = 1.239841984e-6
    //  e * lambda            e
    contrib *= lightParams.photonYield * energyContrib * 1.239841983e-6 / wavelength;
    #else
    contrib *= lightParams.photonYield * energyContrib;
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
