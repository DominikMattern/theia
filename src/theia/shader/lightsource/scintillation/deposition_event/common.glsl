#ifndef _INCLUDE_LIGHTSOURCE_SCINTILLATION_DEPOSITIONEVENT_COMMON
#define _INCLUDE_LIGHTSOURCE_SCINTILLATION_DEPOSITIONEVENT_COMMON

uniform LightParams {
    vec3 startPosition[BATCH_SIZE];
    vec3 endPosition[BATCH_SIZE];

    uint mediumIdx;

    float startTime[BATCH_SIZE];
    float startVelocity[BATCH_SIZE];
    float endVelocity[BATCH_SIZE];

    float cumulativePhotonNumber[BATCH_SIZE];

    float timeConstant;
} lightParams;

#endif
