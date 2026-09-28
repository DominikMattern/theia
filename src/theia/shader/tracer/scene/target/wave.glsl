#ifndef _INCLUDE_TRACER_SCENE_TARGET_WAVE
#define _INCLUDE_TRACER_SCENE_TARGET_WAVE

/**
 * Per-dispatch state of the wavefront variant of the scene target tracer, see
 * `WavefrontSceneForwardTracer`.
 *
 * The tracer splits the trace loop over several dispatches ("waves"). Each wave
 * runs a bounded number of iterations and appends the paths that are still alive
 * to a queue, densely packed via an atomic counter. The next wave is then
 * dispatched with exactly that number of rays (`traceRaysIndirect` reads the
 * counter as a `VkTraceRaysIndirectCommandKHR`), so its warps are filled with
 * long-lived paths instead of one survivor dragging 31 idle lanes along.
 *
 * A dispatch cannot derive which wave it is from the rays it processes - it has
 * to know the source queue before it can load the first record. That single
 * piece of per-dispatch state lives here; the host rebinds this block before
 * recording each wave (the descriptor state is snapshotted per command).
 */

#include "util/buffers.glsl"

/**
 * No explicit binding number, like everywhere else in theia: glslang assigns one
 * per shader (setAutoMapBindings), and hephaistos' CompilerSession then rewrites
 * the numbers so that a block of a given NAME ends up at the same binding in
 * every shader it compiled - which is what lets a ray tracing pipeline merge the
 * bindings of all its shaders into one layout.
 *
 * The invariant to respect is therefore not a number but a scope: every shader
 * of a pipeline must come from the SAME CompilerSession. A second session starts
 * with an empty name -> binding map and renumbers from scratch, which collides
 * with the shaders of the first one.
 */
uniform WaveParams {
    //Queue this dispatch reads its rays from. Unused by the first wave, which
    //samples the light source instead.
    uvec2 srcQueue;
    //Queue the survivors of this dispatch are appended to.
    uvec2 dstQueue;
    //Address of the counter of dstQueue - like every buffer reference in theia
    //this is a 64 bit device address packed into a uvec2, see
    //GL_EXT_buffer_reference_uvec2. The three uints it points at are a
    //VkTraceRaysIndirectCommandKHR, i.e. (count, 1, 1): this shader only ever
    //touches the first of them via Counter(), while the host hands the same
    //address to traceRaysIndirect, whose command processor reads all three as
    //the launch dimensions of the next wave.
    uvec2 dstCounter;
    //Number of trace iterations this dispatch performs before handing the
    //survivors over. Uniform, hence so is the trace loop's bound.
    uint iterations;
    //Number of slots per queue. The ray states are stored as a structure of
    //arrays, so this is also the stride between consecutive fields.
    uint capacity;
    //Non-zero for the wave that samples the light source, zero for the ones
    //continuing the paths out of srcQueue.
    uint firstWave;
} waveParams;

#endif
