#ifndef _INCLUDE_UTIL_OFFSET
#define _INCLUDE_UTIL_OFFSET

//Tunable magnitude of the self-intersection offset applied by offsetRay().
//All three can be overridden from the host by injecting the corresponding
//preprocessor define (e.g. via createPreamble).
#ifndef RAY_OFFSET_SCALE
#define RAY_OFFSET_SCALE 256.0
#endif

#ifndef RAY_OFFSET_ORIGIN
#define RAY_OFFSET_ORIGIN (1.0 / 65536.0)
#endif

#ifndef RAY_OFFSET_THRESHOLD
#define RAY_OFFSET_THRESHOLD (1.0 / 32.0)
#endif
#include "util/float.glsl"

/**
 * Offsets ray position from surface hits to prevent self-intersection, i.e.
 * this ensures that after transmission/reflection the ray is actually on the
 * correct side of the geometry by compensating for the finite numerical
 * precision while introducing minimal bias.
 *
 * Normal (n) points outwards for rays existing the surface, else is flipped.
 * Offsets in normal direction.
 *
 * Taken from Ray Tracing Gems: Chapter 6
 * C. Waechter and N. Binder (2019): "A Fast and Robust Method for Avoiding
 * Self-Intersection"
 *
 * Three constants control the magnitude of the offset:
 *
 *   RAY_OFFSET_SCALE   (default 256.0)
 *       Offset in ULP applied for coordinates at or above the threshold.
 *   RAY_OFFSET_ORIGIN (default 1/65536)
 *       Absolute offset applied for coordinates below the threshold.
 *   RAY_OFFSET_THRESHOLD (default 1/32)
 *       |coordinate| at which the two branches meet.
 *
 * The defaults are the values from the reference publication and are known to
 * be a robust safeguard against self-intersection.
*/
vec3 offsetRay(vec3 p, vec3 n) {
    ivec3 of_i = ivec3(RAY_OFFSET_SCALE * n);

    vec3 p_i = vec3(
        intBitsToFloat(floatBitsToInt(p.x)+((p.x < 0.0) ? -of_i.x : of_i.x)),
        intBitsToFloat(floatBitsToInt(p.y)+((p.y < 0.0) ? -of_i.y : of_i.y)),
        intBitsToFloat(floatBitsToInt(p.z)+((p.z < 0.0) ? -of_i.z : of_i.z))
    );

    return vec3(
        abs(p.x) < RAY_OFFSET_THRESHOLD ? p.x + RAY_OFFSET_ORIGIN * n.x : p_i.x,
        abs(p.y) < RAY_OFFSET_THRESHOLD ? p.y + RAY_OFFSET_ORIGIN * n.y : p_i.y,
        abs(p.z) < RAY_OFFSET_THRESHOLD ? p.z + RAY_OFFSET_ORIGIN * n.z : p_i.z
    );
}

/**
 * Offsets point along normal by given offset
*/
vec3 offsetRay(vec3 p, vec3 n, float o) {
    precise vec3 r = fmaKHR(vec3(o), n, p);
    return r;
}

#endif
