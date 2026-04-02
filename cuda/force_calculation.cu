#include <cuda_runtime.h>
#include "../include/quadtree.h"
#include "../include/body.h"

// =====================================================================
// Kernel 5: Force Calculation (Barnes-Hut Tree Traversal)
// =====================================================================

__global__ void kernelCalculateForces(
    const float* __restrict__ pos_x, const float* __restrict__ pos_y,
    const float* __restrict__ mass, float* __restrict__ acc_x, float* __restrict__ acc_y,
    const QuadNode* __restrict__ nodes, int n, float theta, float G, float softeningSq)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Load body position
    float px = pos_x[i];
    float py = pos_y[i];
    
    // Accumulators for acceleration
    float ax = 0.0f;
    float ay = 0.0f;

    // Start at the root node (index 0)
    int curr = 0; 
    
    // Stackless traversal: we follow the 'next' pointer or descend into 'children'
    while (curr != -1) {
        QuadNode node = nodes[curr];
        
        float dx = node.com_x - px;
        float dy = node.com_y - py;
        float distSq = dx * dx + dy * dy + softeningSq;
        float dist = sqrtf(distSq);
        
        // MAC (Multipole Acceptance Criterion): 
        // We can treat this node as a single distant heavy body if:
        // 1. It is a leaf node (no children to open)
        // 2. OR it is far enough away relative to its size (size / distance < theta)
        if (node.children == 0 || (node.size / dist) < theta) {
            
            if (node.children == 0) { 
                // Leaf Node: To be completely accurate and avoid self-interaction, 
                // we iterate over all actual bodies within this leaf.
                for (int b = node.body_start; b < node.body_start + node.body_count; b++) {
                    if (b != i) { // Do not attract yourself!
                        float b_dx = pos_x[b] - px;
                        float b_dy = pos_y[b] - py;
                        float b_distSq = b_dx * b_dx + b_dy * b_dy + softeningSq;
                        float invDist = rsqrtf(b_distSq);
                        float invDist3 = invDist * invDist * invDist;
                        
                        ax += G * mass[b] * b_dx * invDist3;
                        ay += G * mass[b] * b_dy * invDist3;
                    }
                }
            } else { 
                // Internal Node: Treat its entire mass as sitting at its Center of Mass.
                float invDist = 1.0f / dist; 
                float invDist3 = invDist * invDist * invDist;
                
                ax += G * node.total_mass * dx * invDist3;
                ay += G * node.total_mass * dy * invDist3;
            }
            
            // Skip checking the children of this node, move to the next branch
            curr = node.next; 
        } else {
            // Node is an internal node and it is too close! 
            // We must "open" it and examine its children in more detail.
            curr = node.children;
        }
    }
    
    acc_x[i] = ax;
    acc_y[i] = ay;
}

// Host launcher
void launchForceCalculation(Bodies& bodies, const Quadtree& tree, float theta, float G, float softening, cudaStream_t stream = 0) {
    if (bodies.count == 0) return;
    
    int blockSize = 256;
    int numBlocks = (bodies.count + blockSize - 1) / blockSize;
    
    float softeningSq = softening * softening;
    
    kernelCalculateForces<<<numBlocks, blockSize, 0, stream>>>(
        bodies.pos_x, bodies.pos_y, bodies.mass, bodies.acc_x, bodies.acc_y,
        tree.nodes, bodies.count, theta, G, softeningSq
    );
}
