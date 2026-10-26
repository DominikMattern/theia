#error "This file is meant as documentation and does not contain valid code!"

//This file documents the API through which the rough surface models interact
//with a micro-facet distribution. A model provides the functions below and is
//prepended to either `surface/dielectric_rough/template.glsl` or
//`surface/metal_rough/template.glsl` during compilation (see theia.surface).
//
//The models are free in how they parametrize themselves: nothing outside assumes 
//a particular material property. Everything a model needs it loads itself in 
//prepareMicrofacet().
//
//sampleMicrofacetNormal() draws a facet and evalMicrofacetDensity() states the
//density it drew it with. Keeping the two consistent is what the whole NEE
//machinery rests on - the surface model has no BSDF of its own, it is
//*defined* by this sampling, and theia tabulates the resulting normalization
//from these very functions (surface/microfacet/normalization.glsl).
//
//All directions are in world space. `rayDir` is the incident ray direction, i.e.
//it points *towards* the surface, opposite to `hit.rayNrm`.


//Structure caching whatever is constant across a single surface interaction, i.e.
//does not depend on the sampled facet. Built once per hit and passed into every
//other function.
struct MicrofacetParams { };


//Fills MicrofacetParams. This is where a model loads its own parameters from the
//material table.
MicrofacetParams prepareMicrofacet(
    vec3 rayDir,                    ///< Incident ray direction
    const SurfaceHit hit            ///< Surface intersection being processed
);


//Samples a micro-facet normal in the hemisphere around `hit.rayNrm`.
//The number of random numbers drawn here has to be reported through
//`SurfaceRNGDraws.prepareSurface` of the surface model using this file.
vec3 sampleMicrofacetNormal(
    const MicrofacetParams params,
    vec3 rayDir,                    ///< Incident ray direction
    const SurfaceHit hit,           ///< Surface intersection being processed
    uint idx, inout uint dim        ///< RNG state
);


//The density with which `sampleMicrofacetNormal` draws `microfacetNormal`,
//measured in solid angle and normalized to one over the upper hemisphere:
//
//    int D_p(h) dw_h = 1
//
//This is the deterministic twin of `sampleMicrofacetNormal` and must match it
//exactly - it is what turns the sampling into a BSDF. For a model that samples
//facets proportional to a normal distribution function `D`, this is
//`D(h)*cos(theta_h)`; for one that samples visible normals it is the visible
//normal distribution. A normalization without a closed form has to be
//precomputed on the host and fetched here (see `gaussianNormalization` in
//theia.microfacet).
//Optional, only required for NEE/MIS.
float evalMicrofacetDensity(
    const MicrofacetParams params,
    vec3 microfacetNormal,          ///< Facet to evaluate
    const SurfaceHit hit,           ///< Surface intersection being processed
    vec3 rayDir                     ///< Incident ray direction
);

//Probability that a facet is accepted for the given outgoing direction, in [0,1].
//The caller compares this against a uniform random number. This can be used to
//include masking effects or a boolean allowed/forbidden decision. The template
//already checks that the ray moves into the proper macroscopic hemisphere, so this
//check is not necessary here. The facet loop renormalizes by the resulting
//acceptance rate, and NEE needs that rate in absolute terms; it is computed on
//the device from this function (surface/microfacet/normalization.glsl).
float microfacetAcceptProb(
    const MicrofacetParams params,
    vec3 dirOut,                    ///< Outgoing direction to test
    vec3 microfacetNormal,          ///< Facet the direction came from
    const SurfaceHit hit,           ///< Surface intersection being processed
    vec3 rayDir                     ///< Incident ray direction
);
