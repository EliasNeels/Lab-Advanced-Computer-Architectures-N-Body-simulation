#include <cuda_runtime.h>
#include <cstdint>
#include <cub/cub.cuh>
#include "../include/body.h"
#include "../include/tree.h"
#include "../include/cuda_utils.h"
#include "../include/BoundingBox.h"
#include "../include/kernels.cuh"
#include <algorithm>

// =====================================================================
// Kernel 2: 3D Morton Code Sort — Z-order curve in 3 dimensions
// =====================================================================

// Expand 10 bits to 30 bits with 2-bit gaps for 3D interleaving
__device__ __forceinline__ uint32_t expandBits3D(uint32_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__device__ __forceinline__ uint32_t morton3D(uint32_t x, uint32_t y, uint32_t z) {
    return expandBits3D(x) | (expandBits3D(y) << 1) | (expandBits3D(z) << 2);
}

__global__ void kernelComputeMortonCodes(const float* __restrict__ pos_x,
                                          const float* __restrict__ pos_y,
                                          const float* __restrict__ pos_z,
                                          int n,
                                          float root_min_x, float root_min_y, float root_min_z,
                                          float size,
                                          uint32_t* __restrict__ mortonCodes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Normalize to [0, 1023] (10 bits per dimension for 3D)
    float nx = (pos_x[i] - root_min_x) / size;
    float ny = (pos_y[i] - root_min_y) / size;
    float nz = (pos_z[i] - root_min_z) / size;
    
    uint32_t ix = (uint32_t)fminf(fmaxf(nx * 1023.0f, 0.0f), 1023.0f);
    uint32_t iy = (uint32_t)fminf(fmaxf(ny * 1023.0f, 0.0f), 1023.0f);
    uint32_t iz = (uint32_t)fminf(fmaxf(nz * 1023.0f, 0.0f), 1023.0f);

    mortonCodes[i] = morton3D(ix, iy, iz);
}

__global__ void kernelInitIndices(int* __restrict__ indices, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        indices[i] = i;
    }
}

__global__ void kernelReorderBodies(const float* __restrict__ src_pos_x,
                                     const float* __restrict__ src_pos_y,
                                     const float* __restrict__ src_pos_z,
                                     const float* __restrict__ src_vel_x,
                                     const float* __restrict__ src_vel_y,
                                     const float* __restrict__ src_vel_z,
                                     const float* __restrict__ src_acc_x,
                                     const float* __restrict__ src_acc_y,
                                     const float* __restrict__ src_acc_z,
                                     const float* __restrict__ src_mass,
                                     const float* __restrict__ src_radius,
                                     float* __restrict__ dst_pos_x,
                                     float* __restrict__ dst_pos_y,
                                     float* __restrict__ dst_pos_z,
                                     float* __restrict__ dst_vel_x,
                                     float* __restrict__ dst_vel_y,
                                     float* __restrict__ dst_vel_z,
                                     float* __restrict__ dst_acc_x,
                                     float* __restrict__ dst_acc_y,
                                     float* __restrict__ dst_acc_z,
                                     float* __restrict__ dst_mass,
                                     float* __restrict__ dst_radius,
                                     const int* __restrict__ sortedIndices,
                                     int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int src = sortedIndices[i];
    dst_pos_x[i]  = src_pos_x[src];
    dst_pos_y[i]  = src_pos_y[src];
    dst_pos_z[i]  = src_pos_z[src];
    dst_vel_x[i]  = src_vel_x[src];
    dst_vel_y[i]  = src_vel_y[src];
    dst_vel_z[i]  = src_vel_z[src];
    dst_acc_x[i]  = src_acc_x[src];
    dst_acc_y[i]  = src_acc_y[src];
    dst_acc_z[i]  = src_acc_z[src];
    dst_mass[i]   = src_mass[src];
    dst_radius[i] = src_radius[src];
}

void launchMortonSort(Bodies& bodies, Bodies& scratch,
                      const BoundingBox* d_bbox,
                      uint32_t* d_mortonKeys, uint32_t* d_mortonKeysOut,
                      int* d_indicesIn, int* d_sortedIndices,
                      void*& d_tempStorage, size_t& tempStorageBytes,
                      cudaStream_t stream) {
    int n = bodies.count;
    if (n == 0) return;

    BoundingBox bbox;
    CUDA_CHECK(cudaMemcpy(&bbox, d_bbox, sizeof(BoundingBox), cudaMemcpyDeviceToHost));

    // Compute cube root size (largest dimension) for uniform 3D grid
    float cx = (bbox.min_x + bbox.max_x) * 0.5f;
    float cy = (bbox.min_y + bbox.max_y) * 0.5f;
    float cz = (bbox.min_z + bbox.max_z) * 0.5f;
    float size = fmaxf(fmaxf(bbox.max_x - bbox.min_x, bbox.max_y - bbox.min_y),
                       bbox.max_z - bbox.min_z) * 1.001f;
    float root_min_x = cx - size * 0.5f;
    float root_min_y = cy - size * 0.5f;
    float root_min_z = cz - size * 0.5f;

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    kernelComputeMortonCodes<<<numBlocks, blockSize, 0, stream>>>(
        bodies.pos_x, bodies.pos_y, bodies.pos_z, n,
        root_min_x, root_min_y, root_min_z, size,
        d_mortonKeys
    );

    kernelInitIndices<<<numBlocks, blockSize, 0, stream>>>(d_indicesIn, n);

    size_t tempBytes = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, tempBytes,
                                     d_mortonKeys, d_mortonKeysOut,
                                     d_indicesIn, d_sortedIndices, n, 0, 30, stream);

    if (tempBytes > tempStorageBytes) {
        if (d_tempStorage) cudaFree(d_tempStorage);
        tempStorageBytes = tempBytes;
        CUDA_CHECK(cudaMalloc(&d_tempStorage, tempStorageBytes));
    }

    cub::DeviceRadixSort::SortPairs(d_tempStorage, tempStorageBytes,
                                     d_mortonKeys, d_mortonKeysOut,
                                     d_indicesIn, d_sortedIndices, n, 0, 30, stream);

    scratch.count = n;
    kernelReorderBodies<<<numBlocks, blockSize, 0, stream>>>(
        bodies.pos_x, bodies.pos_y, bodies.pos_z,
        bodies.vel_x, bodies.vel_y, bodies.vel_z,
        bodies.acc_x, bodies.acc_y, bodies.acc_z,
        bodies.mass, bodies.radius,
        scratch.pos_x, scratch.pos_y, scratch.pos_z,
        scratch.vel_x, scratch.vel_y, scratch.vel_z,
        scratch.acc_x, scratch.acc_y, scratch.acc_z,
        scratch.mass, scratch.radius,
        d_sortedIndices, n
    );

    // Swap pointers (O(1), zero GPU cost)
    std::swap(bodies.pos_x, scratch.pos_x);
    std::swap(bodies.pos_y, scratch.pos_y);
    std::swap(bodies.pos_z, scratch.pos_z);
    std::swap(bodies.vel_x, scratch.vel_x);
    std::swap(bodies.vel_y, scratch.vel_y);
    std::swap(bodies.vel_z, scratch.vel_z);
    std::swap(bodies.acc_x, scratch.acc_x);
    std::swap(bodies.acc_y, scratch.acc_y);
    std::swap(bodies.acc_z, scratch.acc_z);
    std::swap(bodies.mass, scratch.mass);
    std::swap(bodies.radius, scratch.radius);
}
