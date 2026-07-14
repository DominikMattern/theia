#ifndef _INCLUDE_SURFACE_MICROFACET_GAUSSIAN
#define _INCLUDE_SURFACE_MICROFACET_GAUSSIAN

#include "math.glsl"

/*
Sampling of micro-facets from a Gaussian slope-angle distribution. The facet polar angle theta 
follows

    p(theta) ~ sin(theta) * exp(-theta^2 / (2*sigma^2)),   theta in [0, pi/2]

with sigma = ROUGHNESS_PARAMETER. This distribution has no closed-form inverse
CDF, so we sample theta by rejection. The target is factored into an analytically 
invertible proposal times a bounded acceptance weight:

    sin(theta) * exp(-theta^2/2sigma^2) = [theta * exp(-theta^2/2sigma^2)] * [sin(theta)/theta]
                                           \_______ Rayleigh(sigma) _______/   \___ <= 1 ___/

The Rayleigh factor is sampled analytically, truncated to [0, pi/2] so the
proposal never leaves the domain, and accepted with probability sin(theta)/theta,
which is <= 1 on [0, pi/2]. The overall acceptance rate is >= 0.81 for any sigma,
so the loop almost always exits on the first iteration.
*/
vec3 sample_microfacet_normal(vec3 rayDir, const SurfaceHit hit, uint idx, inout uint dim){

    float sigma = loadMaterialConstant(ROUGHNESS_PARAMETER, hit.materialIdx);
    //truncation mass of the Rayleigh proposal on [0, pi/2]
    float c = 1.0 - exp(-(PI_OVER_TWO * PI_OVER_TWO) / (2.0 * sigma * sigma));

    //rejection sample theta from the truncated Rayleigh proposal; accept ~ sin(theta)/theta.
    //4 attempts give a fall-through probability < 1.3e-3 even in the worst case (sigma -> inf),
    //and effectively zero for realistic roughnesses.
    float theta = 0.0;
    for (int i = 0; i < 4; ++i) {
        vec2 rdm = random2D(idx, dim);
        //truncated Rayleigh inverse CDF
        theta = sigma * sqrt(-2.0 * log(1.0 - c * rdm.x));
        float accept = theta > 1e-6 ? sin(theta) / theta : 1.0;
        if (rdm.y < accept)
            break;
    }
    float phi = TWO_PI * random(idx, dim);

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
