#ifndef _INCLUDE_LOOKUP_WRITE
#define _INCLUDE_LOOKUP_WRITE

#include "lookup.glsl"

/*
Writable aliases of the look up table types. Same memory layout as `Table1D` and
`Table2D`, so a compute shader can fill a table whose storage was already
reserved on the host - `TableProperty` allocates it inside the material table
and the shader writes the samples in place afterwards. That is what lets a table
be computed from values that only exist once the material table is complete.

Note that `nx` resp. `nu`/`nv` are the *unpadded* sample counts, while `samples`
carries a one element border on every side (`Table.PADDING`) that the
interpolation stencils read. A writer therefore has to cover
`(nx + 2)` resp. `(nu + 2) * (nv + 2)` entries, row stride `nv + 2`.
*/

layout(buffer_reference, scalar, buffer_reference_align=32) buffer Table1DWrite {
    vec2 range;
    InterpolationMode mode;

    int nx;
    float samples[];
};

layout(buffer_reference, scalar, buffer_reference_align=32) buffer Table2DWrite {
    vec2 range_u;
    vec2 range_v;
    InterpolationMode mode;

    int nu, nv;
    float samples[];
    //memory layout is equivalent to float[nu + 2][nv + 2]
};

#define loadMaterialSlotIdx_Table1DWrite(slotIdx, idx) \
    Table1DWrite(materialTable.data[materialTable.stride * (slotIdx) + (idx)])
#define loadMaterialSlotIdx_Table2DWrite(slotIdx, idx) \
    Table2DWrite(materialTable.data[materialTable.stride * (slotIdx) + (idx)])

/**
 * Maps a padded sample index to the coordinate it represents. Border entries
 * are clamped onto the edge of the range, i.e. constant extrapolation, which is
 * exactly what linear interpolation needs at the boundary.
 */
float tableCoordAt(int paddedIdx, int n, vec2 range) {
    int i = clamp(paddedIdx - 1, 0, n - 1);
    return n > 1 ? mix(range.x, range.y, float(i) / float(n - 1)) : range.x;
}

#endif
