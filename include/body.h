#pragma once

#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
#include "cuda_utils.h"

struct Bodies {
    // Device arrays (each N elements)
    float* pos_x;      // positions
    float* pos_y;
    float* pos_z;
    float* vel_x;      // velocities
    float* vel_y;
    float* vel_z;
    float* acc_x;      // accelerations
    float* acc_y;
    float* acc_z;
    float* mass;        // mass
    float* radius;      // collision radius = cbrt(mass)
    int    count;       // number of active bodies
    int    capacity;    // allocated capacity
};

struct BodyHost {
    float px, py, pz;
    float vx, vy, vz;
    float mass;
    float radius;
};

// Allocate bodies on device
inline void bodiesAllocDevice(Bodies& b, int capacity) {
    b.capacity = capacity;
    b.count = 0;
    CUDA_CHECK(cudaMalloc(&b.pos_x, capacity * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.pos_y, capacity * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.pos_z, capacity * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.vel_x, capacity * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.vel_y, capacity * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.vel_z, capacity * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.acc_x, capacity * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.acc_y, capacity * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.acc_z, capacity * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.mass,  capacity * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.radius, capacity * sizeof(float)));
}

// Free device bodies
inline void bodiesFreeDevice(Bodies& b) {
    if(b.pos_x) cudaFree(b.pos_x); if(b.pos_y) cudaFree(b.pos_y); if(b.pos_z) cudaFree(b.pos_z);
    if(b.vel_x) cudaFree(b.vel_x); if(b.vel_y) cudaFree(b.vel_y); if(b.vel_z) cudaFree(b.vel_z);
    if(b.acc_x) cudaFree(b.acc_x); if(b.acc_y) cudaFree(b.acc_y); if(b.acc_z) cudaFree(b.acc_z);
    if(b.mass)  cudaFree(b.mass);  if(b.radius) cudaFree(b.radius);
    b.count = 0;
    b.capacity = 0;
}

// Upload N bodies from host array to device (appending at b.count)
inline void bodiesUpload(Bodies& b, const BodyHost* hostBodies, int n) {
    if (b.count + n > b.capacity) {
        fprintf(stderr, "bodiesUpload: capacity exceeded (%d + %d > %d)\n",
                b.count, n, b.capacity);
        return;
    }
    int off = b.count;
    // Temporary staging arrays
    float* hpx = (float*)malloc(n * sizeof(float));
    float* hpy = (float*)malloc(n * sizeof(float));
    float* hpz = (float*)malloc(n * sizeof(float));
    float* hvx = (float*)malloc(n * sizeof(float));
    float* hvy = (float*)malloc(n * sizeof(float));
    float* hvz = (float*)malloc(n * sizeof(float));
    float* hm  = (float*)malloc(n * sizeof(float));
    float* hr  = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        hpx[i] = hostBodies[i].px;
        hpy[i] = hostBodies[i].py;
        hpz[i] = hostBodies[i].pz;
        hvx[i] = hostBodies[i].vx;
        hvy[i] = hostBodies[i].vy;
        hvz[i] = hostBodies[i].vz;
        hm[i]  = hostBodies[i].mass;
        hr[i]  = hostBodies[i].radius;
    }
    CUDA_CHECK(cudaMemcpy(b.pos_x + off, hpx, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b.pos_y + off, hpy, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b.pos_z + off, hpz, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b.vel_x + off, hvx, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b.vel_y + off, hvy, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b.vel_z + off, hvz, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b.mass  + off, hm,  n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b.radius+ off, hr,  n * sizeof(float), cudaMemcpyHostToDevice));
    // Zero out accelerations for new bodies
    CUDA_CHECK(cudaMemset(b.acc_x + off, 0, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(b.acc_y + off, 0, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(b.acc_z + off, 0, n * sizeof(float)));
    free(hpx); free(hpy); free(hpz); free(hvx); free(hvy); free(hvz); free(hm); free(hr);
    b.count += n;
}