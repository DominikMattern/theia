#error "This file is meant as documentation and does not contain valid code!"

//This file documents the API through which the rough surface models interact
//with a micro-facet distribution. A model provides the five functions below and
//is prepended to either `surface/dielectric_rough/template.glsl` or
//`surface/metal_rough/template.glsl` during compilation (see theia.surface).
//
//The models are intentionally free in how they parametrize themselves: nothing
//outside assumes a particular material property, let alone a single scalar
//roughness. Everything a model needs it loads itself in prepare_microfacet().
//
//Three of the functions come in pairs: sample_microfacet_normal() draws a facet
//and eval_microfacet_density() states the density it drew it with;
//check_microfacet() is not part of the API, its deterministic twin
//microfacet_accept_prob() is. Keeping the pairs consistent is what the whole
//NEE machinery rests on - the surface model has no BSDF of its own, it is
//*defined* by this sampling, and theia tabulates the resulting normalization
//from these very functions (surface/microfacet/normalization.glsl).
//
//All directions are in world space. `rayDir` is the incident ray direction, i.e.
//it points *towards* the surface, opposite to `hit.rayNrm`.

/**
 * MANDATORY. Structure caching whatever is constant across a single surface
 * interaction, i.e. does not depend on the sampled facet. Built once per hit and
 * passed into every other function, so loop invariant work - loading material
 * properties, building a tangent basis - happens once rather than per attempt.
*/
struct MicrofacetParams { };

/**
 * MANDATORY. Fills MicrofacetParams. This is where a model loads its own
 * parameters from the material table; the amount and meaning of those is
 * entirely up to the model.
*/
MicrofacetParams prepare_microfacet(
    vec3 rayDir,                    ///< Incident ray direction
    const SurfaceHit hit            ///< Surface intersection being processed
);

/**
 * MANDATORY. Samples a micro-facet normal in the hemisphere around
 * `hit.rayNrm`. May itself use rejection internally.
 *
 * Note
 * ----
 * The number of random numbers drawn here has to be reported through
 * `SurfaceRNGDraws.prepareSurface` of the surface model using this file.
*/
vec3 sample_microfacet_normal(
    const MicrofacetParams params,
    vec3 rayDir,                    ///< Incident ray direction
    const SurfaceHit hit,           ///< Surface intersection being processed
    uint idx, inout uint dim        ///< RNG state
);

/**
 * MANDATORY. The density with which `sample_microfacet_normal` draws
 * `microfacetNormal`, measured in solid angle and normalized to one over the
 * upper hemisphere:
 *
 *     int D_p(h) dw_h = 1
 *
 * This is the deterministic twin of `sample_microfacet_normal` and must match it
 * exactly - it is what turns the sampling into a BSDF. For a model that samples
 * facets proportional to a normal distribution function `D`, this is
 * `D(h)*cos(theta_h)`; for one that samples visible normals it is the visible
 * normal distribution.
 *
 * A normalization without a closed form has to be precomputed on the host and
 * fetched here (see `gaussianNormalization` in theia.microfacet). Since this is
 * only called on NEE paths, such a fetch does not belong in MicrofacetParams.
 *
 * OPTIONAL. Required for NEE/MIS.
*/
float eval_microfacet_density(
    const MicrofacetParams params,
    vec3 microfacetNormal,          ///< Facet to evaluate
    const SurfaceHit hit,           ///< Surface intersection being processed
    vec3 rayDir                     ///< Incident ray direction
);

/**
 * MANDATORY. Probability that a facet is accepted for the given outgoing
 * direction, in [0,1]. The caller turns it into a decision by comparing against
 * a single uniform, so a model needing no randomness simply returns 0.0 or 1.0.
 *
 * This is where a model states its own validity criterion: that the facet is hit
 * from the front, that the outgoing direction is not masked by the micro
 * structure, or anything else. The macroscopic hemisphere is *not* checked here -
 * the surface template does that.
 *
 * Note
 * ----
 * Whatever this returns, the facet loop renormalizes by the resulting acceptance
 * rate, and NEE needs that rate in absolute terms. It is computed on the device
 * from this function, so no separate bookkeeping is required - but a model whose
 * acceptance is expensive makes the whole loop expensive, since it dominates the
 * cost of a surface hit.
*/
float microfacet_accept_prob(
    const MicrofacetParams params,
    vec3 dirOut,                    ///< Outgoing direction to test
    vec3 microfacetNormal,          ///< Facet the direction came from
    const SurfaceHit hit,           ///< Surface intersection being processed
    vec3 rayDir                     ///< Incident ray direction
);
