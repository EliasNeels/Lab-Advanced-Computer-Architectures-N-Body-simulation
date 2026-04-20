#include <cuda_runtime.h>
#include <cstdint>
#include "../include/tree.h"
#include "../include/body.h"
#include "../include/cuda_utils.h"
#include "../include/kernels.cuh"
#include "../include/BoundingBox.h"

// =====================================================================
// Kernel 3 & 4: GPU Octree Construction & Center of Mass (BFS Layer Method)
// =====================================================================

__global__ void kernelInitRoot(OctNode* nodes, float cx, float cy, float cz, float size, int n) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        OctNode& root = nodes[0];
        root.center_x = cx;
        root.center_y = cy;
        root.center_z = cz;
        root.size = size;
        root.com_x = 0; root.com_y = 0; root.com_z = 0;
        root.total_mass = 0;
        root.children = 0;
        root.body_start = 0;
        root.body_count = n;
        root.next = -1;
    }
}

__global__ void kernelSubdivide(OctNode* nodes, int processedNodes, int numNodesToProcess, 
                                int* nodeCount, const uint32_t* mortonCodes, int level, int leafCapacity) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numNodesToProcess) return;
    
    int ni = processedNodes + i;
    OctNode& node = nodes[ni];
    
    // Stop subdividing at max capacity or max depth (10 levels for 30-bit 3D morton)
    if (node.body_count <= leafCapacity || level >= 10) {
        node.children = 0;
        return;
    }
    
    int start = node.body_start;
    int end = start + node.body_count;
    
    // 3D: each level uses 3 bits (octant index 0-7)
    int shift = 27 - 3 * level;
    
    // Find 7 split points dividing bodies into 8 octants
    int splits[8];
    splits[0] = start;
    
    for (int s = 1; s < 8; s++) {
        int l = (s == 1) ? start : splits[s-1];
        int r = end;
        while (l < r) {
            int m = l + (r - l) / 2;
            if (((mortonCodes[m] >> shift) & 7) < (uint32_t)s)
                l = m + 1;
            else
                r = m;
        }
        splits[s] = l;
    }
    
    // Allocate 8 contiguous children atomically
    int childBase = atomicAdd(nodeCount, 8);
    node.children = childBase;
    
    // 8 octant offsets: (x,y,z) combinations of ±0.5
    float offX[8] = {-0.5f, 0.5f, -0.5f, 0.5f, -0.5f, 0.5f, -0.5f, 0.5f};
    float offY[8] = {-0.5f, -0.5f, 0.5f, 0.5f, -0.5f, -0.5f, 0.5f, 0.5f};
    float offZ[8] = {-0.5f, -0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    
    float halfSize = node.size * 0.5f;
    
    for (int q = 0; q < 8; q++) {
        OctNode& child = nodes[childBase + q];
        child.center_x = node.center_x + offX[q] * halfSize;
        child.center_y = node.center_y + offY[q] * halfSize;
        child.center_z = node.center_z + offZ[q] * halfSize;
        child.size = halfSize;
        child.body_start = splits[q];
        child.body_count = ((q < 7) ? splits[q+1] : end) - splits[q];
        // Next pointer: last child exits to parent's next, others chain to next sibling
        child.next = (q < 7) ? (childBase + q + 1) : node.next;
        child.children = 0;
        child.com_x = 0; child.com_y = 0; child.com_z = 0;
        child.total_mass = 0;
    }
}

__global__ void kernelCenterOfMass(OctNode* nodes, const float* pos_x, const float* pos_y, const float* pos_z,
                                   const float* mass, int startIdx, int numNodesToProcess) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numNodesToProcess) return;
    
    int ni = startIdx + i;
    OctNode& node = nodes[ni];
    
    if (node.children == 0) { 
        // Leaf node
        float mx = 0, my = 0, mz = 0, tm = 0;
        for (int b = node.body_start; b < node.body_start + node.body_count; b++) {
            float m = mass[b];
            mx += pos_x[b] * m;
            my += pos_y[b] * m;
            mz += pos_z[b] * m;
            tm += m;
        }
        if (tm > 0) {
            node.com_x = mx / tm;
            node.com_y = my / tm;
            node.com_z = mz / tm;
        } else {
            node.com_x = node.center_x;
            node.com_y = node.center_y;
            node.com_z = node.center_z;
        }
        node.total_mass = tm;
    } else { 
        // Internal node — pull from 8 children
        float mx = 0, my = 0, mz = 0, tm = 0;
        int childBase = node.children;
        for (int q = 0; q < 8; q++) {
            OctNode& child = nodes[childBase + q];
            float c_mass = child.total_mass;
            if (c_mass > 0) {
                mx += child.com_x * c_mass;
                my += child.com_y * c_mass;
                mz += child.com_z * c_mass;
                tm += c_mass;
            }
        }
        if (tm > 0) {
            node.com_x = mx / tm;
            node.com_y = my / tm;
            node.com_z = mz / tm;
        } else {
            node.com_x = node.center_x;
            node.com_y = node.center_y;
            node.com_z = node.center_z;
        }
        node.total_mass = tm;
    }
}

void launchBuildTree(Octree& tree, Bodies& bodies, const BoundingBox* d_bbox, const uint32_t* d_mortonCodes,
                     int leafCapacity, cudaStream_t stream) {
    int n = bodies.count;
    if (n == 0) return;

    int maxNodes = 8 * n + 1024;
    if (tree.nodes == nullptr || maxNodes > tree.maxNodes) {
        if(tree.nodes != nullptr) {
            cudaFree(tree.nodes);
            cudaFree(tree.nodeCount);
        }
        tree.maxNodes = maxNodes;
        CUDA_CHECK(cudaMalloc(&tree.nodes, maxNodes * sizeof(OctNode)));
        CUDA_CHECK(cudaMalloc(&tree.nodeCount, sizeof(int)));
    }

    BoundingBox h_bbox;
    CUDA_CHECK(cudaMemcpyAsync(&h_bbox, d_bbox, sizeof(BoundingBox), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream); 
    
    float cx = (h_bbox.min_x + h_bbox.max_x) * 0.5f;
    float cy = (h_bbox.min_y + h_bbox.max_y) * 0.5f;
    float cz = (h_bbox.min_z + h_bbox.max_z) * 0.5f;
    float size = fmaxf(fmaxf(h_bbox.max_x - h_bbox.min_x, h_bbox.max_y - h_bbox.min_y),
                       h_bbox.max_z - h_bbox.min_z) * 1.001f;

    int initialCount = 1;
    CUDA_CHECK(cudaMemcpyAsync(tree.nodeCount, &initialCount, sizeof(int), cudaMemcpyHostToDevice, stream));

    kernelInitRoot<<<1, 1, 0, stream>>>(tree.nodes, cx, cy, cz, size, n);
    
    int levelBoundaries[32];
    levelBoundaries[0] = 0;
    levelBoundaries[1] = 1;

    int processedNodes = 0;
    int currentNodeCount = 1;
    int currentLevel = 0;

    while (processedNodes < currentNodeCount && currentLevel < 10) {
        int numNodesToProcess = currentNodeCount - processedNodes;
        int blockSize = 256;
        int numBlocks = (numNodesToProcess + blockSize - 1) / blockSize;
        
        kernelSubdivide<<<numBlocks, blockSize, 0, stream>>>(
            tree.nodes, processedNodes, numNodesToProcess, tree.nodeCount,
            d_mortonCodes, currentLevel, leafCapacity
        );
        
        processedNodes += numNodesToProcess;
        currentLevel++;
        
        CUDA_CHECK(cudaMemcpyAsync(&currentNodeCount, tree.nodeCount, sizeof(int), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);
        
        levelBoundaries[currentLevel + 1] = currentNodeCount;
    }

    // Bottom-up center of mass
    for (int l = currentLevel; l >= 0; l--) {
        int startIdx = levelBoundaries[l];
        int numNodesToProcess = levelBoundaries[l+1] - startIdx;
        
        if (numNodesToProcess > 0) {
            int blockSize = 256;
            int numBlocks = (numNodesToProcess + blockSize - 1) / blockSize;
            kernelCenterOfMass<<<numBlocks, blockSize, 0, stream>>>(
                tree.nodes, bodies.pos_x, bodies.pos_y, bodies.pos_z, bodies.mass,
                startIdx, numNodesToProcess
            );
        }
    }
}
