#ifndef _INCLUDE_LIGHTSOURCE_VOXEL_FORWARD
#define _INCLUDE_LIGHTSOURCE_VOXEL_FORWARD

#include "math.glsl"
#include "util/sample.glsl"

/**
 * Isotropic light source sampling positions uniformly inside the cells of a
 * regular grid spanning an axis aligned box. Consecutive blocks of
 * raysPerCell rays are assigned to consecutive cells starting at cellOffset.
 * Cells are enumerated in C order, i.e. the last axis is the fastest.
 *
 * If LIGHT_SOURCE_CHECK_MEDIUM is defined, each candidate position is checked
 * to lie inside the medium the source emits into by querying the scene. Up to
 * LIGHT_SOURCE_MAX_ATTEMPTS candidates are tried, after which the sample gets
 * rejected by returning a zero direction. The number of created rays per cell
 * is counted.
 * This requires the trace params of the scene tracer.
 */

uniform LightParams {
    vec3 lower;
    uint mediumIdx;

    vec3 cellSize;
    uint raysPerCell;

    uvec3 gridShape;
    uint cellOffset;

    uvec2 countAdr;
    float t_min;
    float t_max;

    float contribFwd;
} lightParams;

#ifdef LIGHT_SOURCE_CHECK_MEDIUM

#ifndef _INCLUDE_TRACER_SCENE_FORWARD_IO
#error "Checking the medium is only supported by scene forward tracers"
#endif

#include "scene/medium.glsl"
#include "util/buffers.glsl"

#ifndef LIGHT_SOURCE_MAX_ATTEMPTS
#define LIGHT_SOURCE_MAX_ATTEMPTS 1
#endif

#endif

ForwardRay sampleLight(uint idx, inout uint dim) {
    //sample wavelength using wavelength source
    #ifdef LIGHT_SOURCE_EMIT_PARTICLE
    float wavelength = sampleWavelength(idx, dim);
    #else
    float contrib;
    float wavelength = sampleWavelength(contrib, idx, dim);
    #endif

    //fetch cell
    uint cellIdx = idx / lightParams.raysPerCell; //relative to this batch
    uint cell = lightParams.cellOffset + cellIdx;
    uvec3 shape = lightParams.gridShape;
    uvec3 ijk = uvec3(cell / (shape.y * shape.z), (cell / shape.z) % shape.y, cell % shape.z);
    vec3 cellLower = vec3(ijk) * lightParams.cellSize + lightParams.lower;

    vec3 position, direction;
    #ifdef LIGHT_SOURCE_CHECK_MEDIUM
    bool valid = false;
    for (uint i = 0; i < LIGHT_SOURCE_MAX_ATTEMPTS && !valid; ++i) {
        vec3 u = vec3(random2D(idx, dim), random(idx, dim));
        position = u * lightParams.cellSize + cellLower;
        //reused as emission direction, see above
        direction = sampleUnitSphere(random2D(idx, dim));
        uint mediumIdx;
        bool found = queryMedium(
            params.tlas, position, direction, params.propagation.maxDist, mediumIdx);
        #ifdef LIGHT_SOURCE_ACCEPT_LOST_RAYS
        valid = !found || mediumIdx == lightParams.mediumIdx;
        #else
        valid = found && mediumIdx == lightParams.mediumIdx;
        #endif
    }
    if (valid) {
        //count created rays
        atomicAdd(UIntBuffer(lightParams.countAdr).values[cellIdx], 1);
    }
    else {
        //reject sample
        direction = vec3(0.0);
    }
    #else
    vec3 u = vec3(random2D(idx, dim), random(idx, dim));
    position = u * lightParams.cellSize + cellLower;
    direction = sampleUnitSphere(random2D(idx, dim));
    #endif

    //sample startTime
    float v = random(idx, dim);
    float time = mix(lightParams.t_min, lightParams.t_max, v);

    //create forward ray
    return createForwardRay(
        position,
        direction,
        wavelength,
        lightParams.mediumIdx,
        time
        #ifndef LIGHT_SOURCE_EMIT_PARTICLE
        , contrib * lightParams.contribFwd
        #endif
    );
}

#endif
