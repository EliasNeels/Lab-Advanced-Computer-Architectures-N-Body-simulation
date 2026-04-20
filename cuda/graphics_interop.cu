#include <cuda_runtime.h>
#include "../include/body.h"
#include "../include/kernels.cuh"

// =====================================================================
// Graphics Interop: 3D VBO Packing
// Layout: [x, y, z, radius] per body (float4)
// =====================================================================

__global__ void kernelPackVBO(const float* __restrict__ pos_x, 
                               const float* __restrict__ pos_y,
                               const float* __restrict__ pos_z,
                               const float* __restrict__ radius, 
                               float4* __restrict__ vbo_data, int n) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        vbo_data[i] = make_float4(pos_x[i], pos_y[i], pos_z[i], radius[i]);
    }
}

extern "C" void launchPackVBO(const Bodies& bodies, float* d_vbo, cudaStream_t stream) {
    if (bodies.count == 0 || d_vbo == nullptr) return;
    int blockSize = 256;
    int numBlocks = (bodies.count + blockSize - 1) / blockSize;
    
    kernelPackVBO<<<numBlocks, blockSize, 0, stream>>>(
        bodies.pos_x, bodies.pos_y, bodies.pos_z, bodies.radius, (float4*)d_vbo, bodies.count
    );
}
