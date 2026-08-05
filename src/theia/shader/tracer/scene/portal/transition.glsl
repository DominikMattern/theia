#ifndef _INCLUDE_PORTAL_TRANSITION
#define _INCLUDE_PORTAL_TRANSITION

/**
 * Portal transition tables + reframe helper (closest-hit stage).
 *
 * A "portal" is a surface carrying MATERIAL_PORTAL_BIT. When the surface model
 * TRANSMITS a ray across it (transmitRay / crossBorder), the photon is switched
 * to another scene / coordinate frame so fine geometry far from the world origin
 * can be traced in LOCAL (small-magnitude) coordinates. See MultiSceneForwardTracer.
 *
 * The crossing is a finite-state machine: the photon carries
 *   frame = ivec2(sceneId, instanceId)   (sceneId 0 = main; instanceId = context).
 * A host-built table maps (sceneId, instanceId, hitPortal) -> (nextScene,
 * nextInstanceId, T), where T is the object->world placement of the ARRIVAL box in
 * the target scene. `portalReframe` re-expresses the (already surface-processed)
 * ray in that frame: object-space hit `objPos` (source placement factored out by
 * the hardware) times the arrival transform T gives small target coordinates, so a
 * matched scale on both boxes cancels and the net map is a rigid isometry.
 *
 * This header is self-contained and included by surface/propagate/forward.glsl
 * under #ifdef RAY_PORTAL, i.e. BEFORE the surface model. It relies on the
 * including translation unit having already declared: SurfaceHit (scene/types.glsl),
 * ForwardRay/TRACE_RAY (ray.glsl), MATERIAL_PORTAL_BIT (material.glsl); and on
 * gl_InstanceID being valid (closest-hit stage). It only compiles under RAY_PORTAL.
 *
 * Table indexing (built host-side, see trace.py): PortalIndexTable / TransitionTable
 * / PortalMetaTable are indexed by sceneId (0-based, main INCLUDED at index 0).
 */

#include "util/offset.glsl"
#include "util/buffers.glsl"

//Per scene (indexed by sceneId, main at 0): device address of an int[] mapping
//gl_InstanceID -> compact portal index within that scene (-1 if not a portal).
readonly buffer PortalIndexTable { uvec2 portalIndexAdr[]; };
//Per scene (indexed by sceneId): device address of a uint[] transition table,
//14 words per record: [0]=nextScene, [1]=nextInstanceId, [2..13]=T (mat4x3,
//column-major, 12 floats). Record index = instanceId * portalCount + portalIdx.
readonly buffer TransitionTable { uvec2 transAdr[]; };
//Per scene (indexed by sceneId): number of portal boxes, i.e. the row stride of
//the transition table.
readonly buffer PortalMetaTable { int portalCount[]; };

//Stride of one transition record, in uint words.
const uint PORTAL_TRANSITION_STRIDE = 14u;

//A single decoded portal transition.
struct PortalTransition {
    int nextScene;
    int nextInstanceId;
    mat4x3 transform;   //object->world placement of the arrival box
};

//Look up the transition for a portal hit in the photon's current frame.
PortalTransition lookupTransition(int sceneId, int instanceId, int glInstanceID) {
    int portalIdx = IntBuffer(portalIndexAdr[sceneId]).values[glInstanceID];
    int stride = portalCount[sceneId];
    uint base = uint(instanceId * stride + portalIdx) * PORTAL_TRANSITION_STRIDE;
    UIntBuffer tb = UIntBuffer(transAdr[sceneId]);

    PortalTransition t;
    t.nextScene = int(tb.values[base + 0]);
    t.nextInstanceId = int(tb.values[base + 1]);
    t.transform = mat4x3(
        uintBitsToFloat(tb.values[base +  2]), uintBitsToFloat(tb.values[base +  3]),
        uintBitsToFloat(tb.values[base +  4]), uintBitsToFloat(tb.values[base +  5]),
        uintBitsToFloat(tb.values[base +  6]), uintBitsToFloat(tb.values[base +  7]),
        uintBitsToFloat(tb.values[base +  8]), uintBitsToFloat(tb.values[base +  9]),
        uintBitsToFloat(tb.values[base + 10]), uintBitsToFloat(tb.values[base + 11]),
        uintBitsToFloat(tb.values[base + 12]), uintBitsToFloat(tb.values[base + 13])
    );
    return t;
}

/**
 * Switch the ray to the portal's target scene/frame AFTER the surface model has
 * transmitted/crossed it. `ray.direction` must already hold the WORLD-space outgoing
 * direction the surface produced (refracted, or unchanged for a border); `ray.mediumIdx`
 * must already be set by the surface. This rewrites position, direction and frame:
 *   - direction is re-expressed in the target frame via the rigid net map
 *     T * worldToObj (rotation only, so the refraction angle is preserved);
 *   - the coarse world position the surface set is DISCARDED in favour of the precise
 *     object-space hit mapped by the arrival transform T (small target coordinates).
 * The caller (surface model) returns RESULT_CODE_VOLUME_HIT so the trace loop
 * re-traces against the newly selected acceleration structure.
 */
void portalReframe(inout ForwardRay ray, const SurfaceHit hit) {
    PortalTransition t = lookupTransition(ray.frame.x, ray.frame.y, gl_InstanceID);
    mat4x3 T = t.transform;
    //direction of travel through the surface (object space); pushes the new origin
    //off the arrival surface into the transmitted side.
    vec3 fwd = hit.inward ? -hit.objNrm : hit.objNrm;
    ray.direction = normalize(mat3(T) * (hit.worldToObj * ray.direction));
    ray.position  = offsetRay(T * vec4(hit.objPos, 1.0), normalize(mat3(T) * fwd));
    ray.frame     = ivec2(t.nextScene, t.nextInstanceId);
}

#endif
