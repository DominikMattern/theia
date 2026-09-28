#ifndef _INCLUDE_RESPONSE_SPECTRUM
#define _INCLUDE_RESPONSE_SPECTRUM

//declares Table1D and lookUp(); the enclosing response does not pull it in
#include "lookup.glsl"

//the block name must differ from the enclosing HitResponse's ResponseParams
uniform ValueParams {
    //using a buffer reference allows us to change it without recreating the pipeline
    Table1D spectrumTableAddress;
} valueParams;


float responseValue(HitItem hit, uint idx, inout uint dim) {
    float efficiency = lookUp(valueParams.spectrumTableAddress, hit.wavelength, 0.0);
    #ifdef RAY_PARTICLE
    float u = random(idx, dim);
    if(u < efficiency){
        return 1.0;
    }
    else{
        return 0.0;
    }
    #else
    return efficiency * hit.contrib;
    #endif
}

#endif