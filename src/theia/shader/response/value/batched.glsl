#ifndef _INCLUDE_RESPONSE_BATCHED
#define _INCLUDE_RESPONSE_BATCHED

#include "util/buffers.glsl"
#include "util/launchid.glsl"

/**
 * Sums up the values of hits per bin and detector. A batch is partitioned into
 * RESPONSE_CELLS_PER_BATCH cells, each owning a consecutive block of paths as
 * given by the running sum of paths per cell. If RESPONSE_BIN_INDEX is defined,
 * each cell maps to an arbitrary bin, otherwise the i-th cell to the i-th bin.
 * Bins are ordered as (bin, detector). If RESPONSE_STORE_SQUARES is defined,
 * the sum of squared values is stored after all bins of the sums.
 */

readonly buffer ResponseParams {
    uvec2 bufferAdr;
    uint bufferCount;
    uint bufferSize;    //total number of entries per buffer
    uint binCount;      //number of bins per detector
    uint detectorCount; //one if integrating all detectors

    //Running sum of the paths each cell owns.
    uint cumulativeCount[RESPONSE_CELLS_PER_BATCH];
#ifdef RESPONSE_BIN_INDEX
    //Bin each cell's hits are summed into
    uint binIndex[RESPONSE_CELLS_PER_BATCH];
#endif
} responseParams;

int findResponseCell(uint idx) {
    int lo = 0;
    int hi = RESPONSE_CELLS_PER_BATCH - 1;

    while (lo < hi)
    {
        int mid = (lo + hi) >> 1;

        if (idx < responseParams.cumulativeCount[mid])
            hi = mid;
        else
            lo = mid + 1;
    }

    return lo;
}

void response(HitItem item, uint idx, inout uint dim) {
    //process hit
    float value = responseValue(item, idx, dim);
    //ignore zero and NaNs
    if (!(value != 0.0)) return;

    //find bin; ignore paths beyond the last cell and bins outside the range
    if (idx >= responseParams.cumulativeCount[RESPONSE_CELLS_PER_BATCH - 1]) return;
    int cell = findResponseCell(idx);
    #ifdef RESPONSE_BIN_INDEX
    uint bin = responseParams.binIndex[cell];
    #else
    uint bin = uint(cell);
    #endif
    if (bin >= responseParams.binCount) return;

    //find detector; ignore detectors outside the range
    #ifdef RESPONSE_INTEGRATE_ALL
    uint det = 0;
    #else
    if (item.objectId < 0 || uint(item.objectId) >= responseParams.detectorCount)
        return;
    uint det = uint(item.objectId);
    #endif
    uint entry = bin * responseParams.detectorCount + det;

    //hash launch id to select a random histogram for storing the value
    //this is to avoid serialization of atomicAdds of the same bin
    uint bufferIdx = getScrambledLaunchId(responseParams.bufferCount);
    uint offset = bufferIdx * responseParams.bufferSize;

    //store value
    BinBuffer bins = BinBuffer(responseParams.bufferAdr);
    atomicAdd(bins.values[offset + entry], value);
    #ifdef RESPONSE_STORE_SQUARES
    uint squaresOffset = responseParams.binCount * responseParams.detectorCount;
    BinScalar square = BinScalar(value) * BinScalar(value);
    atomicAdd(bins.values[offset + squaresOffset + entry], square);
    #endif
}

#endif
