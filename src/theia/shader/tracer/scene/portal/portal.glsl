#ifndef _INCLUDE_TRACER_SCENE_PORTAL
#define _INCLUDE_TRACER_SCENE_PORTAL

/**
 * Portal / multi-scene support for the (forward) scene target tracer: the parts
 * that are NOT the transition itself.
 *
 * A "portal" is a surface carrying MATERIAL_PORTAL_BIT. The actual scene switch
 * (transform position/direction, update frame) now happens inside the surface
 * transmit path (surface/propagate/forward.glsl -> portalReframe, defined in
 * tracer/scene/portal/transition.glsl), so a portal can be coupled to any surface
 * model. This header provides the two pieces that live outside that path:
 *   - portalSelectTlas(): acceleration-structure selection for the raygen (always
 *     available; outside PORTAL_HIT_STAGE),
 *   - portalObjectId(): per-scene detector objectId for the closest-hit (needs the
 *     bound ObjectIdMap; define PORTAL_HIT_STAGE before including).
 *
 * The photon carries frame = ivec2(sceneId, instanceId) (sceneId 0 = main).
 *
 * Table indexing (built host-side, see trace.py): SubTlasTable / SubObjectIdTable
 * are indexed by (sceneId - 1) (main excluded; main uses params.tlas / the bound
 * ObjectIdMap).
 */

#include "util/offset.glsl"
#include "util/buffers.glsl"

//--- sub-scene tables (bound by block name, like ObjectIdMap) ----------------
//Sub-scene TLAS device addresses, indexed by (sceneId - 1); sceneId 0 = main
//(which uses params.tlas). std430 uvec2 array -> no std140 stride pitfall.
//Needed by the raygen (AS selection), hence outside PORTAL_HIT_STAGE.
readonly buffer SubTlasTable { uvec2 subTlas[]; };

//Select the acceleration structure address for the photon's current frame.
//sceneId 0 -> main TLAS (params.tlas); else the sub-scene's TLAS.
uvec2 portalSelectTlas(ivec2 frame) {
    return frame.x == 0 ? params.tlas : subTlas[frame.x - 1];
}

#ifdef PORTAL_HIT_STAGE

//Device address of each sub-scene's ObjectIdMap tensor, indexed by (sceneId-1).
readonly buffer SubObjectIdTable { uvec2 subObjectIdAdr[]; };

//objectId for the current hit, reading the right per-scene ObjectIdMap.
int portalObjectId(ivec2 frame) {
    if (frame.x == 0)
        return objectIdMap[gl_InstanceID];   //main scene: the bound ObjectIdMap
    return IntBuffer(subObjectIdAdr[frame.x - 1]).values[gl_InstanceID];
}

#endif //PORTAL_HIT_STAGE

#endif
