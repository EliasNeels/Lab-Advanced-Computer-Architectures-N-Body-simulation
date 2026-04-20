#include <cuda_runtime.h>
#include <cstdint>
#include <cub/cub.cuh>
#include "../include/body.h"
#include "../include/quadtree.h"
#include "../include/cuda_utils.h"
#include "../include/BoundingBox.h"
#include "../include/kernels.cuh"
#include <algorithm> // for std::swap


// =====================================================================
// Kernel 2: Morton Code Sort — Spatial hashing for cache-friendly layout
// =====================================================================

// Interleave bits for 2D Morton code (Z-order curve)
__device__ __forceinline__ uint32_t expandBits(uint32_t v) {
    v = (v | (v << 16)) & 0x0000FFFF;
    v = (v | (v <<  8)) & 0x00FF00FF;
    v = (v | (v <<  4)) & 0x0F0F0F0F;
    v = (v | (v <<  2)) & 0x33333333;
    v = (v | (v <<  1)) & 0x55555555;
    return v;
}

__device__ __forceinline__ uint32_t morton2D(uint32_t x, uint32_t y) {
    return expandBits(x) | (expandBits(y) << 1);
}

// Compute Morton codes from a perfectly squared bounding box
__global__ void kernelComputeMortonCodes(const float* __restrict__ pos_x,
                                          const float* __restrict__ pos_y,
                                          int n,
                                          float root_min_x, float root_min_y,
                                          float size,
                                          uint32_t* __restrict__ mortonCodes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Normalize to [0, 65535] using the SQUARE tree size, clamping out-of-bounds
    float nx = (pos_x[i] - root_min_x) / size;
    float ny = (pos_y[i] - root_min_y) / size;
    
    uint32_t ix = (uint32_t)fminf(fmaxf(nx * 65535.0f, 0.0f), 65535.0f);
    uint32_t iy = (uint32_t)fminf(fmaxf(ny * 65535.0f, 0.0f), 65535.0f);

    mortonCodes[i] = morton2D(ix, iy);
}

// Kernel to initialize an array with 0, 1, 2, ..., n-1
__global__ void kernelInitIndices(int* __restrict__ indices, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        indices[i] = i;
    }
}

// Kernel to reorder body arrays by sorted indices
__global__ void kernelReorderBodies(const float* __restrict__ src_pos_x,
                                     const float* __restrict__ src_pos_y,
                                     const float* __restrict__ src_vel_x,
                                     const float* __restrict__ src_vel_y,
                                     const float* __restrict__ src_acc_x,
                                     const float* __restrict__ src_acc_y,
                                     const float* __restrict__ src_mass,
                                     const float* __restrict__ src_radius,
                                     float* __restrict__ dst_pos_x,
                                     float* __restrict__ dst_pos_y,
                                     float* __restrict__ dst_vel_x,
                                     float* __restrict__ dst_vel_y,
                                     float* __restrict__ dst_acc_x,
                                     float* __restrict__ dst_acc_y,
                                     float* __restrict__ dst_mass,
                                     float* __restrict__ dst_radius,
                                     const int* __restrict__ sortedIndices,
                                     int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int src = sortedIndices[i];
    dst_pos_x[i]  = src_pos_x[src];
    dst_pos_y[i]  = src_pos_y[src];
    dst_vel_x[i]  = src_vel_x[src];
    dst_vel_y[i]  = src_vel_y[src];
    dst_acc_x[i]  = src_acc_x[src];
    dst_acc_y[i]  = src_acc_y[src];
    dst_mass[i]   = src_mass[src];
    dst_radius[i] = src_radius[src];
}

// Host launcher: compute morton codes, sort, reorder bodies
// ALL scratch buffers are pre-allocated by the caller — zero allocations here!
void launchMortonSort(Bodies& bodies, Bodies& scratch,
                      const BoundingBox* d_bbox,
                      uint32_t* d_mortonKeys, uint32_t* d_mortonKeysOut,
                      int* d_indicesIn, int* d_sortedIndices,
                      void*& d_tempStorage, size_t& tempStorageBytes,
                      cudaStream_t stream) {
    int n = bodies.count;
    if (n == 0) return;

    // Read bounding box from device
    BoundingBox bbox;
    CUDA_CHECK(cudaMemcpy(&bbox, d_bbox, sizeof(BoundingBox), cudaMemcpyDeviceToHost));

    // EXACT SAME math as the Quadtree root to ensure perfect geometric alignment
    float cx = (bbox.min_x + bbox.max_x) * 0.5f;
    float cy = (bbox.min_y + bbox.max_y) * 0.5f;
    float size = fmaxf(bbox.max_x - bbox.min_x, bbox.max_y - bbox.min_y) * 1.001f;
    float root_min_x = cx - size * 0.5f;
    float root_min_y = cy - size * 0.5f;

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Step 1: Compute Morton codes using square alignment
    kernelComputeMortonCodes<<<numBlocks, blockSize, 0, stream>>>(
        bodies.pos_x, bodies.pos_y, n,
        root_min_x, root_min_y, size,
        d_mortonKeys
    );

    // Step 2: Initialize indices array [0, 1, 2, ..., n-1] (pre-allocated buffer)
    kernelInitIndices<<<numBlocks, blockSize, 0, stream>>>(d_indicesIn, n);

    // Step 3: CUB radix sort (key-value pairs: morton code -> body index)
    size_t tempBytes = 0;
    
    // First call to determine temporary storage requirements
    cub::DeviceRadixSort::SortPairs(nullptr, tempBytes,
                                     d_mortonKeys, d_mortonKeysOut,
                                     d_indicesIn, d_sortedIndices, n, 0, 32, stream);

    // Only reallocate if we need more space (should only happen once at startup)
    if (tempBytes > tempStorageBytes) {
        if (d_tempStorage) cudaFree(d_tempStorage);
        tempStorageBytes = tempBytes;
        CUDA_CHECK(cudaMalloc(&d_tempStorage, tempStorageBytes));
    }

    // Second call to actually sort
    cub::DeviceRadixSort::SortPairs(d_tempStorage, tempStorageBytes,
                                     d_mortonKeys, d_mortonKeysOut,
                                     d_indicesIn, d_sortedIndices, n, 0, 32, stream);

    // Step 4: Reorder bodies from 'bodies' into 'scratch' using sorted indices
    scratch.count = n;
    kernelReorderBodies<<<numBlocks, blockSize, 0, stream>>>(
        bodies.pos_x, bodies.pos_y, bodies.vel_x, bodies.vel_y,
        bodies.acc_x, bodies.acc_y, bodies.mass, bodies.radius,
        scratch.pos_x, scratch.pos_y, scratch.vel_x, scratch.vel_y,
        scratch.acc_x, scratch.acc_y, scratch.mass, scratch.radius,
        d_sortedIndices, n
    );

    // Step 5: Swap pointers — O(1), zero GPU cost!
    // 'scratch' now holds sorted data, so swap all array pointers.
    // After this, 'bodies' points to sorted data, 'scratch' becomes the reusable buffer.
    std::swap(bodies.pos_x, scratch.pos_x);
    std::swap(bodies.pos_y, scratch.pos_y);
    std::swap(bodies.vel_x, scratch.vel_x);
    std::swap(bodies.vel_y, scratch.vel_y);
    std::swap(bodies.acc_x, scratch.acc_x);
    std::swap(bodies.acc_y, scratch.acc_y);
    std::swap(bodies.mass, scratch.mass);
    std::swap(bodies.radius, scratch.radius);
    // capacity stays the same on both, count is updated
}
