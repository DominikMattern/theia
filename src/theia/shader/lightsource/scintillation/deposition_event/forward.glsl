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

    //sample direction
    vec3 direction = sampleUnitSphere(random2D(idx, dim));

    //sample scintillation time delay from exponential decay
    float v = random(idx, dim);
    float total_time = time - log(1.0 - v) * lightParams.timeConstant;
    
    //calculate contribution
    #ifndef LIGHT_SOURCE_EMIT_PARTICLE
    #ifdef SCINTILLATION_USE_ENERGY
    // To get from photon count to energy in eV, multiply by
    //    h * c             h * c
    // ------------  with  -------- = 1.239841984e-6
    //  e * lambda            e
    contrib *= lightParams.photonYield * energyContrib * 1.239841983e-6 / wavelength;
    #else
    contrib *= lightParams.photonYield * energyContrib;
    #endif
    #endif

    //create forward ray
    return createForwardRay(
        position,
        direction,
        wavelength,
        lightParams.mediumIdx,
        total_time
        #ifndef LIGHT_SOURCE_EMIT_PARTICLE
        , contrib
        #endif
    );
}

#endif
