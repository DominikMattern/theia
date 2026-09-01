#ifndef _INCLUDE_LIGHTSOURCE_SCINTILLATION_DEPOSITIONEVENT_COMMON
#define _INCLUDE_LIGHTSOURCE_SCINTILLATION_DEPOSITIONEVENT_COMMON

readonly buffer LightParams {
    vec3 startPosition[EVENTS_PER_BATCH];
    vec3 endPosition[EVENTS_PER_BATCH];

    uint mediumIdx[EVENTS_PER_BATCH];

    float startTime[EVENTS_PER_BATCH];
    float startVelocity[EVENTS_PER_BATCH];
    float endVelocity[EVENTS_PER_BATCH];

    float timeConstant[EVENTS_PER_BATCH];

#if defined(LIGHT_SOURCE_EMIT_PARTICLE) || defined(LIGHT_SOURCE_DETERMINISTIC_RAYS)
    //Running sum of the number of paths each element owns.
    uint cumulativeCount[EVENTS_PER_BATCH];
#else
    //Running sum of the expected photon number.
    float cumulativePhotonNumber[EVENTS_PER_BATCH];
#endif

#ifdef LIGHT_SOURCE_DETERMINISTIC_RAYS
    // expected photons / numRays.
    float photonsPerRay[EVENTS_PER_BATCH];
#endif
} lightParams;

/**
 * Returns the batch element the given path belongs to, by binary search over the
 * number of paths each element owns.
 */
#if defined(LIGHT_SOURCE_EMIT_PARTICLE) || defined(LIGHT_SOURCE_DETERMINISTIC_RAYS)
int findBatchElement(uint idx) {
    int lo = 0;
    int hi = EVENTS_PER_BATCH - 1;

    while (lo < hi)
    {
        int mid = (lo + hi) >> 1;

        if (idx < lightParams.cumulativeCount[mid])
            hi = mid;
        else
            lo = mid + 1;
    }

    return lo;
}
#endif

#endif
