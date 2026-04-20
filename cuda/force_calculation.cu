#include <cuda_runtime.h>
#include "../include/tree.h"
#include "../include/body.h"
#include "../include/kernels.cuh"

// =====================================================================
// Kernel 5: 3D Force Calculation (Barnes-Hut Octree Traversal)
//           + Analytic Dark Matter Halo (Logarithmic Potential)
// =====================================================================

__global__ void kernelCalculateForces(
    const float* __restrict__ pos_x, const float* __restrict__ pos_y, const float* __restrict__ pos_z,
    const float* __restrict__ mass, 
    float* __restrict__ acc_x, float* __restrict__ acc_y, float* __restrict__ acc_z,
    const OctNode* __restrict__ nodes, int n, float theta, float G, float softeningSq,
    float haloVcSq, float haloCoreSq)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float px = pos_x[i];
    float py = pos_y[i];
    float pz = pos_z[i];
    
    float ax = 0.0f, ay = 0.0f, az = 0.0f;

    int curr = 0;
    
    while (curr != -1) {
        OctNode node = nodes[curr];
        
        float dx = node.com_x - px;
        float dy = node.com_y - py;
        float dz = node.com_z - pz;
        float distSq = dx * dx + dy * dy + dz * dz + softeningSq;
        float dist = sqrtf(distSq);
        
        if (node.children == 0 || (node.size / dist) < theta) {
            if (node.children == 0) { 
                for (int b = node.body_start; b < node.body_start + node.body_count; b++) {
                    if (b != i) {
                        float b_dx = pos_x[b] - px;
                        float b_dy = pos_y[b] - py;
                        float b_dz = pos_z[b] - pz;
                        float b_distSq = b_dx * b_dx + b_dy * b_dy + b_dz * b_dz + softeningSq;
                        float invDist = rsqrtf(b_distSq);
                        float invDist3 = invDist * invDist * invDist;
                        
                        ax += G * mass[b] * b_dx * invDist3;
                        ay += G * mass[b] * b_dy * invDist3;
                        az += G * mass[b] * b_dz * invDist3;
                    }
                }
            } else { 
                float invDist = 1.0f / dist;
                float invDist3 = invDist * invDist * invDist;
                
                ax += G * node.total_mass * dx * invDist3;
                ay += G * node.total_mass * dy * invDist3;
                az += G * node.total_mass * dz * invDist3;
            }
            curr = node.next;
        } else {
            curr = node.children;
        }
    }
    
    // Dark matter halo (logarithmic potential, centered at origin)
    // Only XY plane — galaxies are thin disks, halo acts in the plane
    float r_sq_halo = px * px + py * py + pz * pz + haloCoreSq;
    ax -= haloVcSq * px / r_sq_halo;
    ay -= haloVcSq * py / r_sq_halo;
    az -= haloVcSq * pz / r_sq_halo;
    
    acc_x[i] = ax;
    acc_y[i] = ay;
    acc_z[i] = az;
}

void launchForceCalculation(Bodies& bodies, const Octree& tree, 
                            float theta, float G, float softening,
                            float haloVcSq, float haloCoreSq,
                            cudaStream_t stream) {
    if (bodies.count == 0) return;
    
    int blockSize = 256;
    int numBlocks = (bodies.count + blockSize - 1) / blockSize;
    
    float softeningSq = softening * softening;
    
    kernelCalculateForces<<<numBlocks, blockSize, 0, stream>>>(
        bodies.pos_x, bodies.pos_y, bodies.pos_z,
        bodies.mass, bodies.acc_x, bodies.acc_y, bodies.acc_z,
        tree.nodes, bodies.count, theta, G, softeningSq,
        haloVcSq, haloCoreSq
    );
}
