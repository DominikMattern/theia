layout(local_size_x = 64) in;

#include "result.glsl"
#include "material.glsl"
#include "scene/types.glsl"
#include "lookup/write.glsl"

//user provided code
#include "ray.glsl"
#include "rng.glsl"
#include "surface.glsl"

/*
Fills the acceptance rate table of a rough surface model.

`prepareSurface()` draws micro-facets until one is accepted, so the density of
the accepted facet is renormalized by the acceptance rate

    Z = int  D_p(h) * a_loop(h)  dw_h

with `D_p` the facet sampling density and `a_loop` the probability that the loop
accepts the facet. Both come from the surface model itself - `roughFacetAcceptance`
is the very function the loop uses - so the table cannot drift away from the code
it normalizes.

One invocation per padded table entry. Border entries evaluate at the clamped
coordinate, i.e. constant extrapolation, which is what linear interpolation needs
at the boundary.
*/

uniform NormalizationParams {
    uint materialIdx;
    uint slot;          ///< index of the table slot to fill
    uint reciprocal;    ///< evaluate at 1/e instead of e (outward direction)
} params;

//Quadrature of the facet integral. The polar angle is warped by THETA_WARP
//because the facet density is peaked at small angles; in the warped variable the
//integrand vanishes like u^5 and is smooth across the whole roughness range.
const int N_THETA = 256;
const int N_PHI = 128;
const float THETA_WARP = 3.0;

float acceptanceRate(float mu_i, float eta) {
    //local frame: macroscopic normal along z, incidence in the xz plane
    vec3 nrm = vec3(0.0, 0.0, 1.0);
    float si = sqrt(max(0.0, 1.0 - mu_i * mu_i));
    vec3 rayDir = vec3(-si, 0.0, -mu_i);

    SurfaceHit hit = SurfaceHit(
        params.materialIdx, 0u, 0u, true,
        vec3(0.0), nrm, 0.0, vec3(0.0), nrm, 0.0, rayDir, mat3(1.0)
    );
    MicrofacetParams mfParams = prepareMicrofacet(rayDir, hit);

    float acc = 0.0;
    for (int it = 0; it < N_THETA; it++) {
        float u = (float(it) + 0.5) / float(N_THETA);
        float theta = PI_OVER_TWO * pow(u, THETA_WARP);
        float dTheta = PI_OVER_TWO * THETA_WARP * pow(u, THETA_WARP - 1.0) / float(N_THETA);
        //factor 2 for the mirrored azimuth half, phi only covers [0, pi]
        float weight = 2.0 * sin(theta) * dTheta * (PI / float(N_PHI));
        float st = sin(theta), ct = cos(theta);

        for (int ip = 0; ip < N_PHI; ip++) {
            float phi = (float(ip) + 0.5) * PI / float(N_PHI);
            vec3 h = vec3(st * cos(phi), st * sin(phi), ct);
            float D = evalMicrofacetDensity(mfParams, h, hit, rayDir);
            acc += D * roughFacetAcceptance(mfParams, rayDir, hit, eta, h) * weight;
        }
    }
    return acc;
}

void main() {
    uint i = gl_GlobalInvocationID.x;

#ifdef ROUGH_SURFACE_HAS_TRANSMISSION

    Table2DWrite tab = loadMaterialSlotIdx_Table2DWrite(params.slot, params.materialIdx);
    uint su = uint(tab.nu) + 2, sv = uint(tab.nv) + 2;
    if (i >= su * sv) return;

    float mu_i = tableCoordAt(int(i / sv), tab.nu, tab.range_u);
    float e = tableCoordAt(int(i % sv), tab.nv, tab.range_v);
    //the axis is the inward ratio n_outside/n_inside; the outward direction sees
    //its reciprocal but shares the axis
    float eta = params.reciprocal != 0 ? 1.0 / e : e;
    tab.samples[i] = acceptanceRate(mu_i, eta);

#else

    Table1DWrite tab = loadMaterialSlotIdx_Table1DWrite(params.slot, params.materialIdx);
    if (i >= uint(tab.nx) + 2) return;

    float mu_i = tableCoordAt(int(i), tab.nx, tab.range);
    //metals have no transmission channel, eta never enters the acceptance
    tab.samples[i] = acceptanceRate(mu_i, 1.0);

#endif
}
