#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "body.h"
#include "tree.h"
#include "BoundingBox.h"


#ifdef __cplusplus
extern "C" {
#endif

// Launchers from .cu files
void launchBoundingBox(const Bodies& bodies, BoundingBox* d_bbox, cudaStream_t stream = 0);

void launchMortonSort(Bodies& bodies, Bodies& scratch,
                      const BoundingBox* d_bbox,
                      uint32_t* d_mortonKeys, uint32_t* d_mortonKeysOut,
                      int* d_indicesIn, int* d_sortedIndices,
                      void*& d_tempStorage, size_t& tempStorageBytes,
                      cudaStream_t stream = 0);

void launchBuildTree(Octree& tree, Bodies& bodies, const BoundingBox* d_bbox, const uint32_t* d_mortonCodes,
                     int leafCapacity, cudaStream_t stream = 0);

void launchForceCalculation(Bodies& bodies, const Octree& tree, float theta, float G, float softening,
                            float haloVcSq, float haloCoreSq, cudaStream_t stream = 0);

void launchLeapfrogKickDrift(Bodies& bodies, float dt, cudaStream_t stream = 0);

void launchLeapfrogKick(Bodies& bodies, float dt, cudaStream_t stream = 0);

void launchCollisionDetection(Bodies& bodies, const Octree& tree, cudaStream_t stream = 0);

void launchPackVBO(const Bodies& bodies, float* d_vbo, cudaStream_t stream = 0);

#ifdef __cplusplus
}
#endif
