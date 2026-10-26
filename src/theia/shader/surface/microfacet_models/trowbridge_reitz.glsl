#ifndef _INCLUDE_SURFACE_MICROFACET_TROWBRIDGE_REITZ
#define _INCLUDE_SURFACE_MICROFACET_TROWBRIDGE_REITZ

#include "math.glsl"

struct MicrofacetParams {
    float alpha;
    vec3 tangent1;
    vec3 tangent2;
};

MicrofacetParams prepareMicrofacet(vec3 rayDir, const SurfaceHit hit){
    MicrofacetParams params;
    params.alpha = loadMaterialConstant(ROUGHNESS_PARAMETER, hit.materialIdx);
    params.tangent1 = perpendicularTo(hit.rayNrm);
    params.tangent2 = crosser(hit.rayNrm, params.tangent1);
    return params;
}

//sampling of micro-facets from the Trowbridge-Reitz (GGX) distribution
vec3 sampleMicrofacetNormal(
    const MicrofacetParams params,
    vec3 rayDir,
    const SurfaceHit hit,
    uint idx, inout uint dim
) {

    //sample angles (theta, phi)
    vec2 rdm = random2D(idx, dim);
    float tan2_theta = params.alpha * params.alpha * rdm.x / (1 - rdm.x);
    float phi = TWO_PI * rdm.y;

    float cos_theta = inversesqrt(1.0 + tan2_theta);
    float sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));

    //rotate surface normal
    return cos_theta * hit.rayNrm
        + sin_theta * cos(phi) * params.tangent1
        + sin_theta * sin(phi) * params.tangent2;
}

#ifdef ROUGH_SURFACE_NEE
//Deterministic twin of sampleMicrofacetNormal(): the density with which it
//draws `microfacetNormal`, measured in solid angle (normalized to one over the
//upper hemisphere). Equals D(h)*cos(theta_h) for the Trowbridge-Reitz NDF.
float evalMicrofacetDensity(
    const MicrofacetParams params,
    vec3 microfacetNormal,
    const SurfaceHit hit,
    vec3 rayDir
) {
    float cos_h = dot(microfacetNormal, hit.rayNrm);
    if (cos_h <= 0.0) return 0.0;
    float cos2 = cos_h * cos_h;
    float tan2 = (1.0 - cos2) / cos2;
    float a2 = params.alpha * params.alpha;
    float d = a2 + tan2;
    return a2 / (PI * cos2 * cos_h * d * d);
}

#endif

float microfacetAcceptProb(
    const MicrofacetParams params,
    vec3 dirOut,
    vec3 microfacetNormal,
    const SurfaceHit hit,
    vec3 rayDir
) {
    //only acccept facets that are hit from the front
    return dot(rayDir, microfacetNormal) < 0.0 ? 1.0 : 0.0;
}

#endif
