#include <cuda_runtime.h>
#include <cstdint>
#include <cub/cub.cuh>
#include "../include/body.h"
#include "../include/quadtree.h"
#include "../include/cuda_utils.h"
#include "../include/BoundingBox.h"


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

// Compute Morton codes from normalized positions
__global__ void kernelComputeMortonCodes(const float* __restrict__ pos_x,
                                          const float* __restrict__ pos_y,
                                          int n,
                                          float min_x, float min_y,
                                          float range_x, float range_y,
                                          uint32_t* __restrict__ mortonCodes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Normalize to [0, 65535]
    float nx = (pos_x[i] - min_x) / fmaxf(range_x, 1e-10f);
    float ny = (pos_y[i] - min_y) / fmaxf(range_y, 1e-10f);
    uint32_t ix = (uint32_t)fminf(nx * 65535.0f, 65535.0f);
    uint32_t iy = (uint32_t)fminf(ny * 65535.0f, 65535.0f);

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
void launchMortonSort(Bodies& bodies, const BoundingBox* d_bbox,
                      uint32_t* d_mortonKeys, int* d_sortedIndices,
                      void* d_tempStorage, size_t& tempStorageBytes,
                      cudaStream_t stream) {
    int n = bodies.count;
    if (n == 0) return;

    // Read bounding box from device
    BoundingBox bbox;
    CUDA_CHECK(cudaMemcpy(&bbox, d_bbox, sizeof(BoundingBox), cudaMemcpyDeviceToHost));

    float range_x = bbox.max_x - bbox.min_x;
    float range_y = bbox.max_y - bbox.min_y;

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Step 1: Compute Morton codes
    kernelComputeMortonCodes<<<numBlocks, blockSize, 0, stream>>>(
        bodies.pos_x, bodies.pos_y, n,
        bbox.min_x, bbox.min_y, range_x, range_y,
        d_mortonKeys
    );

    // Step 2: Initialize indices array [0, 1, 2, ..., n-1] on device
    int* d_indicesIn;
    uint32_t* d_keysOut;
    CUDA_CHECK(cudaMalloc(&d_indicesIn, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_keysOut, n * sizeof(uint32_t)));

    kernelInitIndices<<<numBlocks, blockSize, 0, stream>>>(d_indicesIn, n);

    // Step 3: CUB radix sort (key-value pairs: morton code -> body index)
    size_t tempBytes = 0;
    
    // First call to determine temporary storage requirements
    cub::DeviceRadixSort::SortPairs(nullptr, tempBytes,
                                     d_mortonKeys, d_keysOut,
                                     d_indicesIn, d_sortedIndices, n, 0, 32, stream);

    if (tempBytes > tempStorageBytes) {
        if (d_tempStorage) cudaFree(d_tempStorage);
        tempStorageBytes = tempBytes;
        CUDA_CHECK(cudaMalloc(&d_tempStorage, tempStorageBytes));
    }

    // Second call to actually sort
    cub::DeviceRadixSort::SortPairs(d_tempStorage, tempStorageBytes,
                                     d_mortonKeys, d_keysOut,
                                     d_indicesIn, d_sortedIndices, n, 0, 32, stream);

    // Step 4: Reorder bodies in-place using double-buffering
    // Allocate temporary body arrays
    Bodies tmp;
    bodiesAllocDevice(tmp, n);
    tmp.count = n;

    kernelReorderBodies<<<numBlocks, blockSize, 0, stream>>>(
        bodies.pos_x, bodies.pos_y, bodies.vel_x, bodies.vel_y,
        bodies.acc_x, bodies.acc_y, bodies.mass, bodies.radius,
        tmp.pos_x, tmp.pos_y, tmp.vel_x, tmp.vel_y,
        tmp.acc_x, tmp.acc_y, tmp.mass, tmp.radius,
        d_sortedIndices, n
    );

    // Swap pointers (tmp now holds sorted data, so we replace the original arrays)
    // First, free the original unsorted data
    bodiesFreeDevice(bodies);
    
    // Then assign the temporary (now sorted) arrays to the original bodies struct
    bodies.pos_x = tmp.pos_x;
    bodies.pos_y = tmp.pos_y;
    bodies.vel_x = tmp.vel_x;
    bodies.vel_y = tmp.vel_y;
    bodies.acc_x = tmp.acc_x;
    bodies.acc_y = tmp.acc_y;
    bodies.mass  = tmp.mass;
    bodies.radius = tmp.radius;
    bodies.capacity = tmp.capacity;
    bodies.count = tmp.count;

    // Clean up temporary allocations (except the ones we just moved to 'bodies')
    cudaFree(d_indicesIn);
    cudaFree(d_keysOut);
}
