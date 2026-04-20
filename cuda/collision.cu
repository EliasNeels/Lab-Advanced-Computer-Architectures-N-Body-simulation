#include <cuda_runtime.h>
#include "../include/tree.h"
#include "../include/body.h"
#include "../include/kernels.cuh"

// =====================================================================
// Kernel 6: 3D Collision Detection & Resolution (Octree Broad-phase)
// =====================================================================

__device__ bool intersectAABBvsSphere(float min_x, float min_y, float min_z,
                                       float max_x, float max_y, float max_z,
                                       float cx, float cy, float cz, float radius) {
    float closestX = fmaxf(min_x, fminf(cx, max_x));
    float closestY = fmaxf(min_y, fminf(cy, max_y));
    float closestZ = fmaxf(min_z, fminf(cz, max_z));

    float dx = cx - closestX;
    float dy = cy - closestY;
    float dz = cz - closestZ;

    return (dx * dx + dy * dy + dz * dz) <= (radius * radius);
}

__global__ void kernelResolveCollisions(
    float* __restrict__ pos_x, float* __restrict__ pos_y, float* __restrict__ pos_z,
    float* __restrict__ vel_x, float* __restrict__ vel_y, float* __restrict__ vel_z,
    const float* __restrict__ mass, const float* __restrict__ radius,
    const OctNode* __restrict__ nodes, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float px = pos_x[i], py = pos_y[i], pz = pos_z[i];
    float m1 = mass[i], r1 = radius[i];
    float restitution = 0.8f;

    int curr = 0;
    while (curr != -1) {
        OctNode node = nodes[curr];
        
        float min_x = node.center_x - node.size;
        float max_x = node.center_x + node.size;
        float min_y = node.center_y - node.size;
        float max_y = node.center_y + node.size;
        float min_z = node.center_z - node.size;
        float max_z = node.center_z + node.size;
        
        if (intersectAABBvsSphere(min_x, min_y, min_z, max_x, max_y, max_z, px, py, pz, r1)) {
            if (node.children == 0) {
                for (int b = node.body_start; b < node.body_start + node.body_count; b++) {
                    if (b > i) {
                        float dx = pos_x[b] - px;
                        float dy = pos_y[b] - py;
                        float dz = pos_z[b] - pz;
                        float distSq = dx * dx + dy * dy + dz * dz;
                        float rSum = r1 + radius[b];
                        
                        if (distSq < rSum * rSum && distSq > 0.00001f) {
                            float dist = sqrtf(distSq);
                            float nx = dx / dist, ny = dy / dist, nz = dz / dist;
                            
                            float dvx = vel_x[b] - vel_x[i];
                            float dvy = vel_y[b] - vel_y[i];
                            float dvz = vel_z[b] - vel_z[i];
                            
                            float vn = dvx * nx + dvy * ny + dvz * nz;
                            
                            if (vn < 0) {
                                float m2 = mass[b];
                                float j = -(1.0f + restitution) * vn / (1.0f / m1 + 1.0f / m2);
                                
                                atomicAdd(&vel_x[i], -(j / m1) * nx);
                                atomicAdd(&vel_y[i], -(j / m1) * ny);
                                atomicAdd(&vel_z[i], -(j / m1) * nz);
                                atomicAdd(&vel_x[b],  (j / m2) * nx);
                                atomicAdd(&vel_y[b],  (j / m2) * ny);
                                atomicAdd(&vel_z[b],  (j / m2) * nz);
                                
                                float overlap = rSum - dist;
                                float totalMass = m1 + m2;
                                float mr1 = m2 / totalMass, mr2 = m1 / totalMass;
                                
                                atomicAdd(&pos_x[i], -nx * overlap * mr1);
                                atomicAdd(&pos_y[i], -ny * overlap * mr1);
                                atomicAdd(&pos_z[i], -nz * overlap * mr1);
                                atomicAdd(&pos_x[b],  nx * overlap * mr2);
                                atomicAdd(&pos_y[b],  ny * overlap * mr2);
                                atomicAdd(&pos_z[b],  nz * overlap * mr2);
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

void launchCollisionDetection(Bodies& bodies, const Octree& tree, cudaStream_t stream) {
    if (bodies.count == 0) return;
    int blockSize = 256;
    int numBlocks = (bodies.count + blockSize - 1) / blockSize;
    
    kernelResolveCollisions<<<numBlocks, blockSize, 0, stream>>>(
        bodies.pos_x, bodies.pos_y, bodies.pos_z,
        bodies.vel_x, bodies.vel_y, bodies.vel_z,
        bodies.mass, bodies.radius, tree.nodes, bodies.count
    );
}
