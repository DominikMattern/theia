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
 *   - portalObjectId(): per-scene AND per-context detector objectId for the
 *     closest-hit (define PORTAL_HIT_STAGE before including).
 *
 * The photon carries frame = ivec2(sceneId, instanceId) (sceneId 0 = main).
 *
 * Table indexing (built host-side, see scene.py): SubTlasTable is indexed by
 * (sceneId - 1) (main excluded; main uses params.tlas), while ObjectIdTable is
 * indexed by sceneId with main INCLUDED at index 0, like PortalIndexTable /
 * TransitionTable / PortalMetaTable.
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

//only the hit stage needs the row strides; the raygen does not declare them
#include "tracer/scene/portal/meta.glsl"

//Per scene (indexed by sceneId, main at 0): device address of an int[] holding
//the detector ids of that scene as a 2D table [context][gl_InstanceID], stored
//row-major with the context as the SLOW axis. Row stride is sceneMeta[].y, the
//instance count. Owned by MultiScene, which sees the complete transition graph
//and hence the context counts.
//
//This replaces the flat ObjectIdMap a plain Scene binds; that declaration is
//compiled out under RAY_PORTAL (see target/traverse.rchit.glsl).
readonly buffer ObjectIdTable { uvec2 objectIdAdr[]; };

//objectId for the current hit in the photon's current scene AND context.
int portalObjectId(ivec2 frame) {
    int idx = frame.y * sceneMeta[frame.x].y + gl_InstanceID;
    return IntBuffer(objectIdAdr[frame.x]).values[idx];
}

#endif //PORTAL_HIT_STAGE

#endif
