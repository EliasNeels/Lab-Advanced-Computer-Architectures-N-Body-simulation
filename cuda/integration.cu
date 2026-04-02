#include <cuda_runtime.h>
#include "../include/body.h"

// =====================================================================
// Kernel 5: Leapfrog Integration (Kick-Drift-Kick, 2nd order symplectic)
// =====================================================================
//
// Leapfrog KDK splits each timestep into 3 phases:
//   1. KICK:  v(n+1/2) = v(n) + a(n) * dt/2       (half velocity update)
//   2. DRIFT: x(n+1)   = x(n) + v(n+1/2) * dt      (full position update)
//   3. [Recompute forces a(n+1) from new positions]
//   4. KICK:  v(n+1)   = v(n+1/2) + a(n+1) * dt/2   (second half velocity update)
//
// This is 2nd-order accurate and time-reversible, meaning:
//   - Orbits stay stable for millions of timesteps (no energy drift)
//   - Euler would cause planets to spiral inward or outward over time

// Phase 1+2: Half-kick velocity, then full-drift position
__global__ void kernelLeapfrogKickDrift(
    float* __restrict__ pos_x, float* __restrict__ pos_y,
    float* __restrict__ vel_x, float* __restrict__ vel_y,
    const float* __restrict__ acc_x, const float* __restrict__ acc_y,
    int n, float halfDt, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // KICK: half-step velocity update using current acceleration
    float vx = vel_x[i] + acc_x[i] * halfDt;
    float vy = vel_y[i] + acc_y[i] * halfDt;
    
    // DRIFT: full-step position update using half-stepped velocity
    pos_x[i] += vx * dt;
    pos_y[i] += vy * dt;
    
    // Store the half-stepped velocity (will be completed after new forces)
    vel_x[i] = vx;
    vel_y[i] = vy;
}

// Phase 4: Second half-kick using newly computed acceleration
__global__ void kernelLeapfrogKick(
    float* __restrict__ vel_x, float* __restrict__ vel_y,
    const float* __restrict__ acc_x, const float* __restrict__ acc_y,
    int n, float halfDt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // KICK: complete the velocity update with new acceleration
    vel_x[i] += acc_x[i] * halfDt;
    vel_y[i] += acc_y[i] * halfDt;
}

// Host launchers
void launchLeapfrogKickDrift(Bodies& bodies, float dt, cudaStream_t stream = 0) {
    if (bodies.count == 0) return;
    int blockSize = 256;
    int numBlocks = (bodies.count + blockSize - 1) / blockSize;
    float halfDt = dt * 0.5f;
    
    kernelLeapfrogKickDrift<<<numBlocks, blockSize, 0, stream>>>(
        bodies.pos_x, bodies.pos_y, bodies.vel_x, bodies.vel_y,
        bodies.acc_x, bodies.acc_y, bodies.count, halfDt, dt
    );
}

void launchLeapfrogKick(Bodies& bodies, float dt, cudaStream_t stream = 0) {
    if (bodies.count == 0) return;
    int blockSize = 256;
    int numBlocks = (bodies.count + blockSize - 1) / blockSize;
    float halfDt = dt * 0.5f;
    
    kernelLeapfrogKick<<<numBlocks, blockSize, 0, stream>>>(
        bodies.vel_x, bodies.vel_y,
        bodies.acc_x, bodies.acc_y, bodies.count, halfDt
    );
}
