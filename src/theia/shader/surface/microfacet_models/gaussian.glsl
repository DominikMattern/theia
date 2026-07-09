#ifndef _INCLUDE_SURFACE_MICROFACET_GAUSSIAN
#define _INCLUDE_SURFACE_MICROFACET_GAUSSIAN

#include "math.glsl"


//sampling of micro-facets from the Gaussian distribution
vec3 sample_microfacet_normal(vec3 rayDir, const SurfaceHit hit, uint idx, inout uint dim){

    //sample angles (theta, phi)
    vec2 rdm = random2D(idx, dim);
    float theta = lookUpMaterialTable1D(PPF, hit.materialIdx, rdm.x, 0.0);
    float phi = TWO_PI * rdm.y;

    //rotate surface normal
    vec3 tangentialVector1 = perpendicularTo(hit.rayNrm);
    vec3 tangentialVector2 = perpendicularTo(hit.rayNrm, tangentialVector1);
    return cos(theta) * hit.rayNrm + sin(theta) * cos(phi) * tangentialVector1 + sin(theta) * sin(phi) * tangentialVector2;
}

//check that ray hits micro-facet from the front
bool check_microfacet(vec3 dirOut, vec3 microfacetNormal, const SurfaceHit hit, vec3 rayDir, uint idx, inout uint dim){
    return (dot(rayDir, microfacetNormal) < 0.0);
}

#endif
