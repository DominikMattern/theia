#ifndef _INCLUDE_WAVELENGTHSOURCE_SPECTRUM_BATCHED
#define _INCLUDE_WAVELENGTHSOURCE_SPECTRUM_BATCHED

#include "lookup.glsl"

readonly buffer WavelengthParams {
    //Percent point function of each batch element's spectrum. Stored as the raw
    //address and turned into a Table1D on use.
    uvec2 spectrumTableAddress[WAVELENGTH_SOURCE_SPECTRA_PER_BATCH];
    //Running sum of the paths each element owns.
    uint cumulativeCount[WAVELENGTH_SOURCE_SPECTRA_PER_BATCH];
} wavelengthParams;

int findWavelengthBatchElement(uint idx) {
    int lo = 0;
    int hi = WAVELENGTH_SOURCE_SPECTRA_PER_BATCH - 1;

    while (lo < hi)
    {
        int mid = (lo + hi) >> 1;

        if (idx < wavelengthParams.cumulativeCount[mid])
            hi = mid;
        else
            lo = mid + 1;
    }

    return lo;
}

float sampleWavelength(uint idx, inout uint dim) {
    int batchIdx = findWavelengthBatchElement(idx);
    Table1D table = Table1D(wavelengthParams.spectrumTableAddress[batchIdx]);
    return lookUp(table, random(idx, dim), 0.0);
}

float sampleWavelength(out float contrib, uint idx, inout uint dim) {
    contrib = 1.0;
    int batchIdx = findWavelengthBatchElement(idx);
    Table1D table = Table1D(wavelengthParams.spectrumTableAddress[batchIdx]);
    return lookUp(table, random(idx, dim), 0.0);
}

#endif
