#ifndef _INCLUDE_TRACER_SCENE_PORTAL_META
#define _INCLUDE_TRACER_SCENE_PORTAL_META

/**
 * Per-scene portal metadata, shared by transition.glsl and portal.glsl.
 *
 * Both headers may end up in the same translation unit (the closest-hit stage
 * pulls in transition.glsl via the surface model and portal.glsl directly),
 * while the raygen stage only sees portal.glsl. The declaration therefore lives
 * here, behind an include guard, instead of in either of them.
 *
 * Indexed by sceneId (0-based, main INCLUDED at index 0), like PortalIndexTable,
 * TransitionTable and ObjectIdTable. Both entries are row strides of a per-scene
 * 2D table whose slow axis is the photon's context (instanceId):
 *   .x = number of portal boxes -> stride of TransitionTable
 *   .y = number of instances    -> stride of ObjectIdTable
 */
readonly buffer PortalMetaTable { ivec2 sceneMeta[]; };

#endif
