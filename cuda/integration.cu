#include <cuda_runtime.h>
#include "../include/body.h"
#include "../include/kernels.cuh"

// =====================================================================
// Kernel 5: 3D Leapfrog Integration (Kick-Drift-Kick)
//           + Fused kernel for combining Kick(end) + KickDrift(start)
// =====================================================================

__global__ void kernelLeapfrogKickDrift(
    float* __restrict__ pos_x, float* __restrict__ pos_y, float* __restrict__ pos_z,
    float* __restrict__ vel_x, float* __restrict__ vel_y, float* __restrict__ vel_z,
    const float* __restrict__ acc_x, const float* __restrict__ acc_y, const float* __restrict__ acc_z,
    const float* __restrict__ mass,
    int n, float halfDt, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    if (mass[i] >= 100000.0f) {
        vel_x[i] = 0.0f; vel_y[i] = 0.0f; vel_z[i] = 0.0f;
        return; 
    }

    float vx = vel_x[i] + acc_x[i] * halfDt;
    float vy = vel_y[i] + acc_y[i] * halfDt;
    float vz = vel_z[i] + acc_z[i] * halfDt;
    
    pos_x[i] += vx * dt;
    pos_y[i] += vy * dt;
    pos_z[i] += vz * dt;
    
    vel_x[i] = vx;
    vel_y[i] = vy;
    vel_z[i] = vz;
}

__global__ void kernelLeapfrogKick(
    float* __restrict__ vel_x, float* __restrict__ vel_y, float* __restrict__ vel_z,
    const float* __restrict__ acc_x, const float* __restrict__ acc_y, const float* __restrict__ acc_z,
    const float* __restrict__ mass,
    int n, float halfDt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    if (mass[i] >= 100000.0f) return;

    vel_x[i] += acc_x[i] * halfDt;
    vel_y[i] += acc_y[i] * halfDt;
    vel_z[i] += acc_z[i] * halfDt;
}

// Fused kernel: final Kick of step N + KickDrift of step N+1
// Combines two velocity reads + two velocity writes into one of each.
// Uses acc_old (from previous force calc) for the final kick,
// and acc_new (from current force calc) for the new kick-drift.
__global__ void kernelLeapfrogFused(
    float* __restrict__ pos_x, float* __restrict__ pos_y, float* __restrict__ pos_z,
    float* __restrict__ vel_x, float* __restrict__ vel_y, float* __restrict__ vel_z,
    const float* __restrict__ acc_old_x, const float* __restrict__ acc_old_y, const float* __restrict__ acc_old_z,
    const float* __restrict__ acc_new_x, const float* __restrict__ acc_new_y, const float* __restrict__ acc_new_z,
    const float* __restrict__ mass,
    int n, float halfDt, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    if (mass[i] >= 100000.0f) {
        vel_x[i] = 0.0f; vel_y[i] = 0.0f; vel_z[i] = 0.0f;
        return;
    }

    // Final kick from previous step's acceleration
    float vx = vel_x[i] + acc_old_x[i] * halfDt;
    float vy = vel_y[i] + acc_old_y[i] * halfDt;
    float vz = vel_z[i] + acc_old_z[i] * halfDt;
    
    // New kick from current acceleration + drift
    vx += acc_new_x[i] * halfDt;
    vy += acc_new_y[i] * halfDt;
    vz += acc_new_z[i] * halfDt;
    
    pos_x[i] += vx * dt;
    pos_y[i] += vy * dt;
    pos_z[i] += vz * dt;
    
    vel_x[i] = vx;
    vel_y[i] = vy;
    vel_z[i] = vz;
}

void launchLeapfrogKickDrift(Bodies& bodies, float dt, cudaStream_t stream) {
    if (bodies.count == 0) return;
    int blockSize = 256;
    int numBlocks = (bodies.count + blockSize - 1) / blockSize;
    float halfDt = dt * 0.5f;
    
    kernelLeapfrogKickDrift<<<numBlocks, blockSize, 0, stream>>>(
        bodies.pos_x, bodies.pos_y, bodies.pos_z,
        bodies.vel_x, bodies.vel_y, bodies.vel_z,
        bodies.acc_x, bodies.acc_y, bodies.acc_z,
        bodies.mass, bodies.count, halfDt, dt
    );
}

void launchLeapfrogKick(Bodies& bodies, float dt, cudaStream_t stream) {
    if (bodies.count == 0) return;
    int blockSize = 256;
    int numBlocks = (bodies.count + blockSize - 1) / blockSize;
    float halfDt = dt * 0.5f;
    
    kernelLeapfrogKick<<<numBlocks, blockSize, 0, stream>>>(
        bodies.vel_x, bodies.vel_y, bodies.vel_z,
        bodies.acc_x, bodies.acc_y, bodies.acc_z,
        bodies.mass, bodies.count, halfDt
    );
}

void launchLeapfrogFused(Bodies& bodies, 
                          float* acc_old_x, float* acc_old_y, float* acc_old_z,
                          float dt, cudaStream_t stream) {
    if (bodies.count == 0) return;
    int blockSize = 256;
    int numBlocks = (bodies.count + blockSize - 1) / blockSize;
    float halfDt = dt * 0.5f;
    
    kernelLeapfrogFused<<<numBlocks, blockSize, 0, stream>>>(
        bodies.pos_x, bodies.pos_y, bodies.pos_z,
        bodies.vel_x, bodies.vel_y, bodies.vel_z,
        acc_old_x, acc_old_y, acc_old_z,
        bodies.acc_x, bodies.acc_y, bodies.acc_z,
        bodies.mass, bodies.count, halfDt, dt
    );
}