#ifndef _INCLUDE_SURFACE_MICROFACET_TROWBRIDGE_REITZ
#define _INCLUDE_SURFACE_MICROFACET_TROWBRIDGE_REITZ

#include "math.glsl"

//sampling of micro-facets from the Trowbridge-Reitz (GGX) distribution
vec3 sample_microfacet_normal(vec3 rayDir, const SurfaceHit hit, uint idx, inout uint dim){

    float alpha = loadMaterialConstant(ROUGHNESS_PARAMETER, hit.materialIdx);

    //sample angles (theta, phi)
    vec2 rdm = random2D(idx, dim);
    float tan2_theta = alpha * alpha * rdm.x / (1 - rdm.x);
    float phi = TWO_PI * rdm.y;

    float cos_theta = sqrt(1.0 / (1.0 + tan2_theta));
    float sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));

    //rotate surface normal
    vec3 tangentialVector1 = perpendicularTo(hit.rayNrm);
    vec3 tangentialVector2 = perpendicularTo(hit.rayNrm, tangentialVector1);
    return cos_theta * hit.rayNrm + sin_theta * cos(phi) * tangentialVector1 + sin_theta * sin(phi) * tangentialVector2;
}

//check that ray hits micro-facet from the front
bool check_microfacet(vec3 dirOut, vec3 microfacetNormal, const SurfaceHit hit, vec3 rayDir, uint idx, inout uint dim){
    return (dot(rayDir, microfacetNormal) < 0.0);
}

#endif
