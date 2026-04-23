#include <cuda_runtime.h>
#include "../include/tree.h"
#include "../include/body.h"
#include "../include/kernels.cuh"

// =====================================================================
// Kernel 5: 3D Force Calculation (Barnes-Hut Octree Traversal)
//           + Analytic Dark Matter Halo (Logarithmic Potential)
//           Optimized: __launch_bounds__ for register allocation
// =====================================================================

__global__ void __launch_bounds__(256, 4) kernelCalculateForces(
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
    
    float thetaSq = theta * theta;
    float ax = 0.0f, ay = 0.0f, az = 0.0f;

    int curr = 0;
    
    while (curr != -1) {
        OctNode node = nodes[curr];
        
        float dx = node.com_x - px;
        float dy = node.com_y - py;
        float dz = node.com_z - pz;
        float distSq = dx * dx + dy * dy + dz * dz + softeningSq;
        
        float sizeSq = node.size * node.size;
        
        bool openNode = (node.children != 0) && (sizeSq >= thetaSq * distSq);
        unsigned activeMask = __activemask();
        int warpOpenNode = __any_sync(activeMask, openNode);
        
        if (warpOpenNode) {
            curr = node.children;
        } else {
            if (node.children == 0) { 
                #pragma unroll 4
                for (int b = node.body_start; b < node.body_start + node.body_count; b++) {
                    float b_dx = __ldg(&pos_x[b]) - px;
                    float b_dy = __ldg(&pos_y[b]) - py;
                    float b_dz = __ldg(&pos_z[b]) - pz;
                    float b_distSq = b_dx * b_dx + b_dy * b_dy + b_dz * b_dz + softeningSq;
                    float b_invDist = rsqrtf(b_distSq);
                    float invDist3 = b_invDist * b_invDist * b_invDist;
                    float final_mass = __ldg(&mass[b]) * invDist3;
                    
                    ax += final_mass * b_dx;
                    ay += final_mass * b_dy;
                    az += final_mass * b_dz;
                }
            } else { 
                float invDist = rsqrtf(distSq);
                float invDist3 = invDist * invDist * invDist;
                float final_mass = node.total_mass * invDist3;
                
                ax += final_mass * dx;
                ay += final_mass * dy;
                az += final_mass * dz;
            }
            curr = node.next;
        }
    }
    
    // Scale accumulated Barnes-Hut forces by G
    ax *= G;
    ay *= G;
    az *= G;
    
    // Dark matter halo (logarithmic potential, centered at origin)
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
