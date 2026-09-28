layout(local_size_x = 64) in;

#include "material.glsl"
#include "lookup.glsl"

layout(scalar) buffer OutBuf { float data[]; } outBuf;

uniform ProbeParams {
    uint materialIdx;
    uint slot;
    uint nu;
    uint nv;    ///< 0 marks a 1D table
} params;

/*
Reads a micro-facet normalization table back through the regular look up path,
sampled exactly at its grid nodes so linear interpolation returns the stored
values unchanged.
*/
void main() {
    uint i = gl_GlobalInvocationID.x;
    uint nv = max(params.nv, 1u);
    if (i >= params.nu * nv) return;

    if (params.nv == 0u) {
        Table1D tab = Table1D(
            materialTable.data[materialTable.stride * params.slot + params.materialIdx]);
        float u = mix(tab.range.x, tab.range.y, float(i) / float(params.nu - 1u));
        outBuf.data[i] = lookUp(tab, u, -1.0);
    }
    else {
        Table2D tab = Table2D(
            materialTable.data[materialTable.stride * params.slot + params.materialIdx]);
        float u = mix(tab.range_u.x, tab.range_u.y,
                      float(i / nv) / float(params.nu - 1u));
        float v = mix(tab.range_v.x, tab.range_v.y,
                      float(i % nv) / float(nv - 1u));
        outBuf.data[i] = lookUp2D(tab, u, v, -1.0);
    }
}
