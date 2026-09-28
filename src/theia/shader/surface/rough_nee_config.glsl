#ifndef _INCLUDE_SURFACE_ROUGH_NEE_CONFIG
#define _INCLUDE_SURFACE_ROUGH_NEE_CONFIG

/*
Decides whether a rough surface model compiles its NEE parts, and does so *before*
the micro-facet model is included - which is why this is a header of its own
rather than a few lines in the templates. The models need it too: everything they
provide only for NEE, such as `eval_microfacet_density`, hangs off it, and so do
the material properties feeding those (see api.glsl in microfacet_models/).

NEE is unavailable when

  - SURFACE_ROUGH_NO_NEE: the user opted out via the model's `supportNEE`,
  - RAY_PARTICLE: particles cannot carry a weight,
  - DISABLE_NEE: the tracer does not do NEE in the first place.

The first comes from the model's own preamble (see theia.surface), the others
from the tracer's - both are prepended before any surface code, so all three are
visible here.

The Geant4 UNIFIED model has no NEE at all and does not use this header; it marks
itself specular directly (see surface/unified/template.glsl).
*/

#ifdef SURFACE_ROUGH_NO_NEE
//tell the tracer to take its specular path instead of MIS
#define SURFACE_MODEL_SPECULAR
#endif

#if !defined(SURFACE_MODEL_SPECULAR) && !defined(RAY_PARTICLE) && !defined(DISABLE_NEE)
#define ROUGH_SURFACE_NEE
#endif

#endif
