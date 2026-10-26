#ifndef _INCLUDE_SCENE_MEDIUM
#define _INCLUDE_SCENE_MEDIUM

#include "material.glsl"

/**
 * Point location: determines the medium containing a given position.
 *
 * Requires GL_EXT_ray_query and GL_EXT_ray_tracing_position_fetch, but no ray
 * tracing pipeline, i.e. can be used from any shader stage including compute.
 */

#ifndef POSITION_FETCH_ENABLED
#error "queryMedium requires GL_EXT_ray_tracing_position_fetch"
#endif

/**
 * Queries the medium at the given position by casting a single ray in the
 * given direction. Returns true and stores the medium on the ray's side of the
 * closest surface in mediumIdx if a surface was hit within tMax. Returns false
 * otherwise, i.e. if the position lies outside of any geometry (or the scene
 * has a hole in the direction of the ray), leaving mediumIdx untouched.
*/
bool queryMedium(
    uvec2 tlas,             ///< Address of the top level acceleration structure
    vec3 position,          ///< Position to query
    vec3 direction,         ///< Direction of the query ray
    float tMax,             ///< Max distance to search for surfaces
    out uint mediumIdx      ///< Medium at position, if successful
) {
    rayQueryEXT rayQuery;
    rayQueryInitializeEXT(
        rayQuery,
        accelerationStructureEXT(tlas),
        gl_RayFlagsOpaqueEXT,   //flags
        0xFF,                   //cull mask
        position,               //origin
        0.0,                    //t_min
        direction,              //direction
        tMax                    //t_max
    );
    //all geometries are opaque, so there are no candidates to confirm
    while (rayQueryProceedEXT(rayQuery)) {}
    if (
        rayQueryGetIntersectionTypeEXT(rayQuery, true) !=
        gl_RayQueryCommittedIntersectionTriangleEXT
    ) return false;

    //fetch hit triangle (object space)
    vec3 positions[3];
    rayQueryGetIntersectionTriangleVertexPositionsEXT(rayQuery, true, positions);
    vec3 e1 = positions[1] - positions[0];
    vec3 e2 = positions[2] - positions[0];
    //same winding order convention as in resolveIntersection()
    #ifndef OUTWARD_FACE_CLOCK_WISE
    vec3 objNrm = cross(e1, e2);
    #else
    vec3 objNrm = cross(e2, e1);
    #endif

    //check orientation: inward if direction and normal in opposite direction
    vec3 objDir = rayQueryGetIntersectionObjectRayDirectionEXT(rayQuery, true);
    bool inward = dot(objDir, objNrm) <= 0.0;

    //the ray comes from the opposite side it is heading to
    uint materialIdx = rayQueryGetIntersectionInstanceCustomIndexEXT(rayQuery, true);
    uint flags;
    queryMaterialSide(materialIdx, !inward, mediumIdx, flags);

    return true;
}

#endif
