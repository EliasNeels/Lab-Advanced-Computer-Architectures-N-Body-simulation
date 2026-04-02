#include <cuda_runtime.h>
#include "../include/body.h"

// =====================================================================
// Graphics Interop Kernel
// =====================================================================

// Packs SoA position and mass into an AoS VBO format for OpenGL
// Layout: [x, y, radius, 0] (float4 for aligned memory access in OpenGL)
__global__ void kernelPackVBO(const float* __restrict__ pos_x, const float* __restrict__ pos_y, 
                              const float* __restrict__ radius, float4* __restrict__ vbo_data, int n) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        vbo_data[i] = make_float4(pos_x[i], pos_y[i], radius[i], 0.0f);
    }
}

extern "C" void launchPackVBO(const Bodies& bodies, float* d_vbo, cudaStream_t stream) {
    if (bodies.count == 0 || d_vbo == nullptr) return;
    int blockSize = 256;
    int numBlocks = (bodies.count + blockSize - 1) / blockSize;
    
    kernelPackVBO<<<numBlocks, blockSize, 0, stream>>>(
        bodies.pos_x, bodies.pos_y, bodies.radius, (float4*)d_vbo, bodies.count
    );
}
