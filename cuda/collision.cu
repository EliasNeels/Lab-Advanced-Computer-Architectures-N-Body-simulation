#include <cuda_runtime.h>
#include "../include/quadtree.h"
#include "../include/body.h"

// =====================================================================
// Kernel 6: Collision Detection & Resolution (Quadtree Broad-phase)
// =====================================================================

// Helper to check if a circle intersects with a rectangle (AABB)
__device__ bool intersectAABBvsCircle(float min_x, float min_y, float max_x, float max_y, 
                                      float cx, float cy, float radius) {
    float closestX = fmaxf(min_x, fminf(cx, max_x));
    float closestY = fmaxf(min_y, fminf(cy, max_y));

    float distanceX = cx - closestX;
    float distanceY = cy - closestY;

    return (distanceX * distanceX + distanceY * distanceY) <= (radius * radius);
}

__global__ void kernelResolveCollisions(
    float* __restrict__ pos_x, float* __restrict__ pos_y,
    float* __restrict__ vel_x, float* __restrict__ vel_y,
    const float* __restrict__ mass, const float* __restrict__ radius,
    const QuadNode* __restrict__ nodes, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float px = pos_x[i];
    float py = pos_y[i];
    float m1 = mass[i];
    float r1 = radius[i];

    // Restitution coefficient (1.0 = perfectly elastic, 0.0 = completely inelastic/sticky)
    float restitution = 0.8f; 

    int curr = 0; // Start at root
    while (curr != -1) {
        QuadNode node = nodes[curr];
        
        float min_x = node.center_x - node.size;
        float max_x = node.center_x + node.size;
        float min_y = node.center_y - node.size;
        float max_y = node.center_y + node.size;
        
        // Broad-phase: Does this node touch the radius of our body?
        if (intersectAABBvsCircle(min_x, min_y, max_x, max_y, px, py, r1)) {
            if (node.children == 0) { // Leaf node
                for (int b = node.body_start; b < node.body_start + node.body_count; b++) {
                    if (b > i) { // Only process pair once (i < b evaluates strictly once per pair)
                        float dx = pos_x[b] - px;
                        float dy = pos_y[b] - py;
                        float distSq = dx * dx + dy * dy;
                        float rSum = r1 + radius[b];
                        
                        // Narrow-phase: Physical circle intersection
                        if (distSq < rSum * rSum && distSq > 0.00001f) {
                            float dist = sqrtf(distSq);
                            float nx = dx / dist;
                            float ny = dy / dist;
                            
                            float vx1 = vel_x[i];
                            float vy1 = vel_y[i];
                            float vx2 = vel_x[b];
                            float vy2 = vel_y[b];
                            
                            // Relative velocity
                            float dvx = vx2 - vx1;
                            float dvy = vy2 - vy1;
                            
                            // Velocity magnitude along the normal
                            float vn = dvx * nx + dvy * ny;
                            
                            // Only resolve if they are moving towards each other
                            if (vn < 0) {
                                float m2 = mass[b];
                                
                                // Impulse scalar (J)
                                float j = -(1.0f + restitution) * vn;
                                j /= (1.0f / m1 + 1.0f / m2);
                                
                                // Velocity deltas
                                float dv1x = -(j / m1) * nx;
                                float dv1y = -(j / m1) * ny;
                                float dv2x =  (j / m2) * nx;
                                float dv2y =  (j / m2) * ny;
                                
                                // atomicAdd prevents race conditions when 3 or more 
                                // bodies collide simultaneously!
                                atomicAdd(&vel_x[i], dv1x);
                                atomicAdd(&vel_y[i], dv1y);
                                atomicAdd(&vel_x[b], dv2x);
                                atomicAdd(&vel_y[b], dv2y);
                                
                                // Positional correction (prevents bodies sinking into each other)
                                float overlap = rSum - dist;
                                float totalMass = m1 + m2;
                                float mRatio1 = m2 / totalMass;
                                float mRatio2 = m1 / totalMass;
                                
                                atomicAdd(&pos_x[i], -nx * overlap * mRatio1);
                                atomicAdd(&pos_y[i], -ny * overlap * mRatio1);
                                atomicAdd(&pos_x[b],  nx * overlap * mRatio2);
                                atomicAdd(&pos_y[b],  ny * overlap * mRatio2);
                            }
                        }
                    }
                }
                curr = node.next;
            } else {
                curr = node.children;
            }
        } else {
            curr = node.next;
        }
    }
}

// Host launcher
void launchCollisionDetection(Bodies& bodies, const Quadtree& tree, cudaStream_t stream = 0) {
    if (bodies.count == 0) return;
    
    int blockSize = 256;
    int numBlocks = (bodies.count + blockSize - 1) / blockSize;
    
    kernelResolveCollisions<<<numBlocks, blockSize, 0, stream>>>(
        bodies.pos_x, bodies.pos_y, bodies.vel_x, bodies.vel_y,
        bodies.mass, bodies.radius, tree.nodes, bodies.count
    );
}
