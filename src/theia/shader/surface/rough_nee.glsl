#ifndef _INCLUDE_SURFACE_ROUGH_NEE
#define _INCLUDE_SURFACE_ROUGH_NEE

/*
Shared NEE/MIS implementation for the rough surface models. Included by
`dielectric_rough/template.glsl` and `metal_rough/template.glsl` after they
defined `SurfaceProperties`, and only when the three functions can actually be
called (see `ROUGH_SURFACE_NEE` in those files).

The including template has to provide:

    float roughFresnel(float cos_i, const SurfaceProperties props)
        Reflectance at a micro-facet hit under `cos_i`.
    float roughNormalization(const RAY, const SurfaceHit, const SurfaceProperties)
        The acceptance rate Z of the facet loop in prepareSurface(), read from
        the table computed on the host (see theia.microfacet).
    ROUGH_NEE_TRANSMISSION
        Defined if the model can transmit; enables the refraction branch.

Both helpers read the optical constants from `SurfaceProperties` rather than the
media tables: they do not depend on the sampled facet, but the functions below
need them a dozen times per surface hit and every table lookup is a chain of
dependent memory accesses.

--------------------------------------------------------------------------------
The BSDF
--------------------------------------------------------------------------------
These models have no independently defined BSDF - it is defined by the sampler.
prepareSurface() draws facets until one is accepted, so the density of the
accepted facet is renormalized by the acceptance rate Z. Undoing the half vector
Jacobian gives, with w_i = -ray.direction, w_o = newDir, h the half vector,
c = dot(w_i,h) and a() = microfacet_accept_prob():

                 D_p(h) * F(c) * a(w_o,h)
    f(w_i->w_o) = ------------------------            (reflection)
                    4c * Z * |dot(w_o,n)|

                 D_p(h_t) * (1-F(c_i)) * a(w_o,h_t)      n_o^2 |c_o|
    f(w_i->w_o) = ---------------------------------- * ---------------  (refraction)
                        Z * |dot(w_o,n)|               (n_i c_i+n_o c_o)^2

This is the same expression for every material flag combination, because
prepareSurface() reproduces the reflect-and-transmit behaviour in all of them.
The eta^2 radiance factor is *not* included here; transmitRay() applies it.

--------------------------------------------------------------------------------
The sampling density
--------------------------------------------------------------------------------
The NEE proposal is deliberately *not* the facet loop: in the weighted flag
modes the loop's direction density carries its own normalization, which would
need a second table. Instead a single, un-rejected facet is drawn, which has the
analytically known density D_p and needs no table at all. MIS permits any
correct density, so this only costs a little efficiency in the phase-sampling
branch: the proposal can produce directions that leave the correct macroscopic
hemisphere, where f evaluates to zero.
*/

//Which channels the material flags allow the surface to emit into
void roughChannels(const SurfaceHit hit, out bool canReflect, out bool canTransmit) {
    canReflect = (hit.flags & NO_REFLECT_BIT) == 0;
    #ifdef ROUGH_NEE_TRANSMISSION
    bool isDetector = (hit.flags & MATERIAL_DETECTOR_BIT) != 0;
    canTransmit = (hit.flags & NO_TRANSMIT_BIT) == 0 && !isDetector;
    #else
    canTransmit = false;
    #endif
}

/*
The reflection channel at `w_o`: the facet that would produce it, its sampling
density, the Fresnel reflectance there and the model's acceptance probability.
Returns false if no facet can produce `w_o` by reflection.

Only ever called for `w_o` above the macroscopic surface: a reflection below it
is not something the surface emits, and `sampleSurfaceScattering` declares such
a sample void rather than producing it.
*/
bool roughReflectionTerm(
    const RAY ray, const SurfaceHit hit, const SurfaceProperties props,
    const MicrofacetParams mfParams,
    vec3 w_o, out float D, out float c, out float F, out float a
) {
    vec3 w_i = -ray.direction;
    vec3 h = w_i + w_o;
    float len = length(h);
    if (len < 1e-9) return false;
    h /= len;
    if (dot(h, hit.rayNrm) <= 0.0) return false;

    c = dot(w_i, h);
    if (c <= 0.0) return false;

    D = eval_microfacet_density(mfParams, h, hit, ray.direction);
    F = roughFresnel(c, props);
    a = microfacet_accept_prob(mfParams, w_o, h, hit, ray.direction);
    return true;
}

#ifdef ROUGH_NEE_TRANSMISSION
/*
The refraction channel at `w_o`, analogous to `roughReflectionTerm`. `jacobian`
is |dw_h/dw_o| of the refraction half vector. Cannot produce directions in the
upper hemisphere: there `w_o.h` comes out positive and the facet is rejected.
*/
bool roughRefractionTerm(
    const RAY ray, const SurfaceHit hit, const SurfaceProperties props,
    const MicrofacetParams mfParams,
    vec3 w_o, out float D, out float c_i, out float F, out float a,
    out float jacobian
) {
    float n_i = props.n_i;
    float n_o = props.n_o;

    vec3 w_i = -ray.direction;
    vec3 h = -(n_i * w_i + n_o * w_o);
    float len = length(h);
    if (len < 1e-9) return false;
    h /= len;
    if (dot(h, hit.rayNrm) < 0.0) h = -h;

    c_i = dot(w_i, h);
    float c_o = dot(w_o, h);
    if (c_i <= 0.0 || c_o >= 0.0) return false;
    float denom = n_i * c_i + n_o * c_o;
    if (abs(denom) < 1e-9) return false;

    jacobian = n_o * n_o * abs(c_o) / (denom * denom);
    D = eval_microfacet_density(mfParams, h, hit, ray.direction);
    F = roughFresnel(c_i, props);
    a = microfacet_accept_prob(mfParams, w_o, h, hit, ray.direction);
    return true;
}
#endif

/*
Density of the proposal at `scatteredDir`.

`sampleSurfaceScattering` declares every sample void whose direction the surface
cannot emit into, so its output only ever lands in the hemisphere matching the
channel it used. Exactly one channel can therefore have produced a given
direction, and the density is that channel's term - no summation. The voided
samples carry the remaining probability mass; MIS handles a technique that
sometimes produces nothing, since it then contributes nothing either.
*/
float roughScatterPdf(
    const RAY ray, const SurfaceHit hit, const SurfaceProperties props,
    vec3 scatteredDir
) {
    bool canReflect, canTransmit;
    roughChannels(hit, canReflect, canTransmit);

    float cos_o = dot(scatteredDir, hit.rayNrm);
    if (cos_o == 0.0) return 0.0;
    if (cos_o > 0.0 ? !canReflect : !canTransmit) return 0.0;

    MicrofacetParams mfParams = prepare_microfacet(ray.direction, hit);
    float D, c, F, a;

    if (cos_o > 0.0) {
        if (!roughReflectionTerm(ray, hit, props, mfParams, scatteredDir, D, c, F, a))
            return 0.0;
        //the proposal only tosses the Fresnel coin if the other channel exists
        float kappa = canTransmit ? F : 1.0;
        return D * kappa / (4.0 * c);
    }

    #ifdef ROUGH_NEE_TRANSMISSION
    float jacobian;
    if (!roughRefractionTerm(ray, hit, props, mfParams, scatteredDir, D, c, F, a, jacobian))
        return 0.0;
    float kappa = canReflect ? (1.0 - F) : 1.0;
    return D * kappa * jacobian;
    #else
    return 0.0;
    #endif
}

/*
The BSDF f(w_i -> w_o). Unlike the proposal density this *is* selected by
hemisphere: the surface never emits a reflection below nor a transmission above
the macroscopic surface, the facet loop rejects those outright. Returns 0.0 if
`scatteredDir` cannot be reached.
*/
float roughBsdf(
    const RAY ray, const SurfaceHit hit, const SurfaceProperties props,
    vec3 scatteredDir
) {
    bool canReflect, canTransmit;
    roughChannels(hit, canReflect, canTransmit);

    float cos_o = dot(scatteredDir, hit.rayNrm);
    if (cos_o == 0.0) return 0.0;
    if (cos_o > 0.0 ? !canReflect : !canTransmit) return 0.0;

    float Z = roughNormalization(ray, hit, props);
    if (Z <= 0.0) return 0.0;

    MicrofacetParams mfParams = prepare_microfacet(ray.direction, hit);
    float D, c, F, a;

    if (cos_o > 0.0) {
        if (!roughReflectionTerm(ray, hit, props, mfParams, scatteredDir, D, c, F, a))
            return 0.0;
        return D * F * a / (4.0 * c * Z * cos_o);
    }

    #ifdef ROUGH_NEE_TRANSMISSION
    float jacobian;
    if (!roughRefractionTerm(ray, hit, props, mfParams, scatteredDir, D, c, F, a, jacobian))
        return 0.0;
    return D * (1.0 - F) * a * jacobian / (Z * (-cos_o));
    #else
    return 0.0;
    #endif
}

float surfaceScatterProb(
    RAY ray,
    const SurfaceHit hit,
    const SurfaceProperties props,
    vec3 scatteredDir
) {
    return roughScatterPdf(ray, hit, props, scatteredDir);
}

vec3 sampleSurfaceScattering(
    const RAY ray,
    const SurfaceHit hit,
    const SurfaceProperties props,
    out float prob,
    uint idx, inout uint dim
) {
    bool canReflect, canTransmit;
    roughChannels(hit, canReflect, canTransmit);

    MicrofacetParams mfParams = prepare_microfacet(ray.direction, hit);
    vec3 h = sample_microfacet_normal(mfParams, ray.direction, hit, idx, dim);
    float u = random(idx, dim);

    prob = 0.0;
    if (!canReflect && !canTransmit)
        return hit.rayNrm;

    //A facet hit from behind would produce a direction the surface never emits
    //into, and `roughScatterPdf` does not account for such facets. Declaring the
    //sample void keeps the reported density exactly the density of what this
    //function produces, which is what MIS needs. The tracer skips samples with
    //zero probability.
    if (dot(ray.direction, h) >= 0.0)
        return hit.rayNrm;

    float cos_i = abs(dot(ray.direction, h));
    float F = roughFresnel(cos_i, props);
    //pick the branch; if only one is allowed it is taken deterministically
    bool doReflect = canTransmit ? (u < F) : true;
    if (!canReflect) doReflect = false;

    vec3 dir;
    if (doReflect) {
        dir = reflect(ray.direction, h);
    }
    else {
        #ifdef ROUGH_NEE_TRANSMISSION
        dir = refract(ray.direction, h, props.n_i / props.n_o);
        if (dir == vec3(0.0)) return hit.rayNrm;   //total internal reflection
        #else
        return hit.rayNrm;
        #endif
    }

    //A facet steep enough sends its reflection below the macroscopic surface,
    //resp. its refraction above it. The surface does not emit into those
    //directions, so the sample is void - reporting it would make the density
    //disagree with what this function actually produces.
    if (doReflect != (dot(dir, hit.rayNrm) > 0.0))
        return hit.rayNrm;

    //Evaluate the density of the direction we just produced. Going through the
    //same function that surfaceScatterProb() uses keeps the two consistent by
    //construction, which is what MIS relies on.
    prob = roughScatterPdf(ray, hit, props, dir);
    return dir;
}

ResultCode surfaceScatterRay(
    inout RAY ray,
    const SurfaceHit hit,
    const SurfaceProperties props,
    vec3 newDir,
    uint idx, inout uint dim
) {
    //the ray has to hit the surface from the front
    if (dot(ray.direction, hit.rayNrm) >= 0.0)
        return RESULT_CODE_RAY_ABSORBED;

    //The material flags must be honoured here and not only in
    //sampleSurfaceInteraction(): the target guide hands us arbitrary directions,
    //including ones through a channel this surface may not use.
    bool canReflect, canTransmit;
    roughChannels(hit, canReflect, canTransmit);
    float cos_o = dot(newDir, hit.rayNrm);
    if (cos_o > 0.0 ? !canReflect : !canTransmit)
        return RESULT_CODE_RAY_ABSORBED;

    //zero for every remaining direction the surface cannot emit into, i.e. a
    //facet the model rejects or one that does not exist at all
    float f = roughBsdf(ray, hit, props, newDir);
    if (f <= 0.0)
        return RESULT_CODE_RAY_ABSORBED;

    ray.lin_contrib *= f;
    if (cos_o > 0.0)
        return reflectRay(ray, hit, newDir);

    #ifdef ROUGH_NEE_TRANSMISSION
    //transmitRay applies the eta^2 radiance factor itself (backward rays carry
    //radiance, forward rays carry flux), so it must not appear in f
    return transmitRay(ray, hit, newDir);
    #else
    return RESULT_CODE_RAY_ABSORBED;
    #endif
}

#endif
