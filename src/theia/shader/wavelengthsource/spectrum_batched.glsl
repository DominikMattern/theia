#ifndef _INCLUDE_WAVELENGTHSOURCE_SPECTRUM_BATCHED
#define _INCLUDE_WAVELENGTHSOURCE_SPECTRUM_BATCHED

#include "lookup.glsl"

readonly buffer WavelengthParams {
    //Percent point function of each batch element's spectrum. Stored as the raw
    //address and turned into a Table1D or Table2D on use.
    uvec2 spectrumTableAddress[WAVELENGTH_SOURCE_SPECTRA_PER_BATCH];
    //Running sum of the paths each element owns.
    uint cumulativeCount[WAVELENGTH_SOURCE_SPECTRA_PER_BATCH];
#ifdef WAVELENGTH_SOURCE_SECOND_AXIS
    //Coordinate on the second table axis, NaN if the element's table is 1D.
    //This lets 1D and 2D spectra share one batch.
    float axisValue[WAVELENGTH_SOURCE_SPECTRA_PER_BATCH];
#endif
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

/**
 * Looks the given quantile up in the batch element's spectrum.
 *
 * With a second axis, a NaN coordinate marks the table as 1D. Note that the
 * coordinate goes on the SLOW axis (u) and the quantile on the fast one (v):
 * Table2D is row major, so every path of an element then reads the same two
 * rows. lookUp2D maps and clamps both axes itself, so a coordinate outside the
 * table's range needs no handling here.
*/
float sampleWavelengthAt(int batchIdx, float quantile) {
#ifdef WAVELENGTH_SOURCE_SECOND_AXIS
    float axis = wavelengthParams.axisValue[batchIdx];
    if (!isnan(axis)) {
        Table2D table = Table2D(wavelengthParams.spectrumTableAddress[batchIdx]);
        return lookUp2D(table, axis, quantile, 0.0);
    }
#endif
    Table1D table = Table1D(wavelengthParams.spectrumTableAddress[batchIdx]);
    return lookUp(table, quantile, 0.0);
}

float sampleWavelength(uint idx, inout uint dim) {
    int batchIdx = findWavelengthBatchElement(idx);
    return sampleWavelengthAt(batchIdx, random(idx, dim));
}

float sampleWavelength(out float contrib, uint idx, inout uint dim) {
    contrib = 1.0;
    int batchIdx = findWavelengthBatchElement(idx);
    return sampleWavelengthAt(batchIdx, random(idx, dim));
}

#endif
