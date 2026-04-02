#include <cuda_runtime.h>
#include <cstdint>
#include "../include/quadtree.h"
#include "../include/body.h"
#include "../include/cuda_utils.h"

// Declaration from bounding_box.cu
struct BoundingBox {
    float min_x, min_y;
    float max_x, max_y;
};

// =====================================================================
// Kernel 3 & 4: GPU Quadtree Construction & Center of Mass (BFS Layer Method)
// =====================================================================

__global__ void kernelInitRoot(QuadNode* nodes, float cx, float cy, float size, int n) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        QuadNode& root = nodes[0];
        root.center_x = cx;
        root.center_y = cy;
        root.size = size;
        root.com_x = 0;
        root.com_y = 0;
        root.total_mass = 0;
        root.children = 0;
        root.body_start = 0;
        root.body_count = n;
        root.next = -1;
    }
}

__global__ void kernelSubdivide(QuadNode* nodes, int processedNodes, int numNodesToProcess, 
                                int* nodeCount, const uint32_t* mortonCodes, int level, int leafCapacity) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numNodesToProcess) return;
    
    int ni = processedNodes + i;
    QuadNode& node = nodes[ni];
    
    // Stop subdividing if we reached max capacity or max depth
    if (node.body_count <= leafCapacity || level >= 15) {
        node.children = 0;
        return;
    }
    
    int start = node.body_start;
    int end = start + node.body_count;
    
    // The current Level looks at a specific 2-bit window in the 32-bit Morton code.
    int shift = 30 - 2 * level;
    
    // Use Binary Search to find the 3 split points dividing the bodies into 4 quadrants.
    // Because bodies are sorted by Morton code, bits [30-2*level] are strictly monotonically non-decreasing.
    int split1 = start;
    int split2 = start;
    int split3 = start;
    
    int l = start, r = end;
    while (l < r) { int m = l + (r - l) / 2; if (((mortonCodes[m] >> shift) & 3) < 1) l = m + 1; else r = m; }
    split1 = l;
    
    l = split1; r = end;
    while (l < r) { int m = l + (r - l) / 2; if (((mortonCodes[m] >> shift) & 3) < 2) l = m + 1; else r = m; }
    split2 = l;
    
    l = split2; r = end;
    while (l < r) { int m = l + (r - l) / 2; if (((mortonCodes[m] >> shift) & 3) < 3) l = m + 1; else r = m; }
    split3 = l;
    
    // Allocate 4 contiguous children atomically
    int childBase = atomicAdd(nodeCount, 4);
    node.children = childBase;
    
    int starts[4] = {start, split1, split2, split3};
    int counts[4] = {split1 - start, split2 - split1, split3 - split2, end - split3};
    
    float halfSize = node.size * 0.5f;
    float offX[4] = {-0.5f, 0.5f, -0.5f, 0.5f};
    float offY[4] = {-0.5f, -0.5f, 0.5f, 0.5f};
    
    // Next pointer: child 3 exits to current node's next pointer.
    int nexts[4] = {childBase + 1, childBase + 2, childBase + 3, node.next};
    
    for (int q = 0; q < 4; q++) {
        QuadNode& child = nodes[childBase + q];
        child.center_x = node.center_x + offX[q] * halfSize;
        child.center_y = node.center_y + offY[q] * halfSize;
        child.size = halfSize;
        child.body_start = starts[q];
        child.body_count = counts[q];
        child.next = nexts[q];
        child.children = 0;
        child.com_x = 0; child.com_y = 0; child.total_mass = 0;
    }
}

__global__ void kernelCenterOfMass(QuadNode* nodes, const float* pos_x, const float* pos_y, const float* mass, 
                                   int startIdx, int numNodesToProcess) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numNodesToProcess) return;
    
    int ni = startIdx + i;
    QuadNode& node = nodes[ni];
    
    if (node.children == 0) { 
        // Leaf node - read body mass directly
        float mx = 0, my = 0, tm = 0;
        for (int b = node.body_start; b < node.body_start + node.body_count; b++) {
            float m = mass[b];
            mx += pos_x[b] * m;
            my += pos_y[b] * m;
            tm += m;
        }
        if (tm > 0) {
            node.com_x = mx / tm;
            node.com_y = my / tm;
        } else {
            node.com_x = node.center_x;
            node.com_y = node.center_y;
        }
        node.total_mass = tm;
    } else { 
        // Internal node - pull mass from children (already computed from previous level pass)
        float mx = 0, my = 0, tm = 0;
        int childBase = node.children;
        for (int q = 0; q < 4; q++) {
            QuadNode& child = nodes[childBase + q];
            float c_mass = child.total_mass;
            if (c_mass > 0) {
                mx += child.com_x * c_mass;
                my += child.com_y * c_mass;
                tm += c_mass;
            }
        }
        if (tm > 0) {
            node.com_x = mx / tm;
            node.com_y = my / tm;
        } else {
            node.com_x = node.center_x;
            node.com_y = node.center_y;
        }
        node.total_mass = tm;
    }
}

// Host launcher combining both Queueing and GPU Execution
void launchBuildTree(Quadtree& tree, Bodies& bodies, const BoundingBox* d_bbox, const uint32_t* d_mortonCodes,
                     int leafCapacity, cudaStream_t stream) {
    int n = bodies.count;
    if (n == 0) return;

    // ensure device memory is big enough
    int maxNodes = 4 * n + 1024;
    if (tree.nodes == nullptr || maxNodes > tree.maxNodes) {
        if(tree.nodes != nullptr) {
            cudaFree(tree.nodes);
            cudaFree(tree.nodeCount);
        }
        tree.maxNodes = maxNodes;
        CUDA_CHECK(cudaMalloc(&tree.nodes, maxNodes * sizeof(QuadNode)));
        CUDA_CHECK(cudaMalloc(&tree.nodeCount, sizeof(int)));
    }

    BoundingBox h_bbox;
    CUDA_CHECK(cudaMemcpyAsync(&h_bbox, d_bbox, sizeof(BoundingBox), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream); 
    
    // slightly expand box to avoid edge precision bugs
    float cx = (h_bbox.min_x + h_bbox.max_x) * 0.5f;
    float cy = (h_bbox.min_y + h_bbox.max_y) * 0.5f;
    float size = fmaxf(h_bbox.max_x - h_bbox.min_x, h_bbox.max_y - h_bbox.min_y) * 1.001f;

    int initialCount = 1;
    CUDA_CHECK(cudaMemcpyAsync(tree.nodeCount, &initialCount, sizeof(int), cudaMemcpyHostToDevice, stream));

    kernelInitRoot<<<1, 1, 0, stream>>>(tree.nodes, cx, cy, size, n);
    
    int levelBoundaries[32]; // Maximum 16 levels needed for 32-bit Morton code
    levelBoundaries[0] = 0;
    levelBoundaries[1] = 1;

    int processedNodes = 0;
    int currentNodeCount = 1;
    int currentLevel = 0;

    // TOP-DOWN Subdivide
    while (processedNodes < currentNodeCount && currentLevel < 16) {
        int numNodesToProcess = currentNodeCount - processedNodes;
        int blockSize = 256;
        int numBlocks = (numNodesToProcess + blockSize - 1) / blockSize;
        
        kernelSubdivide<<<numBlocks, blockSize, 0, stream>>>(
            tree.nodes, processedNodes, numNodesToProcess, tree.nodeCount,
            d_mortonCodes, currentLevel, leafCapacity
        );
        
        processedNodes += numNodesToProcess;
        currentLevel++;
        
        // Read back the new node count
        CUDA_CHECK(cudaMemcpyAsync(&currentNodeCount, tree.nodeCount, sizeof(int), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream); // Wait for the queue update
        
        levelBoundaries[currentLevel + 1] = currentNodeCount;
    }

    // BOTTOM-UP Center of Mass
    for (int l = currentLevel; l >= 0; l--) {
        int startIdx = levelBoundaries[l];
        int numNodesToProcess = levelBoundaries[l+1] - startIdx;
        
        if (numNodesToProcess > 0) {
            int blockSize = 256;
            int numBlocks = (numNodesToProcess + blockSize - 1) / blockSize;
            kernelCenterOfMass<<<numBlocks, blockSize, 0, stream>>>(
                tree.nodes, bodies.pos_x, bodies.pos_y, bodies.mass,
                startIdx, numNodesToProcess
            );
        }
    }
}
