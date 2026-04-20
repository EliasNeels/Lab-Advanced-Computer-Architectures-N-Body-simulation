#pragma once

#include <vector>
#include <string>
#include <cmath>
#include "body.h"

// =====================================================================
// Simulation Scenario System
// Each scenario defines: physics parameters + initial body configuration
// =====================================================================

struct SimulationConfig {
    std::string name;
    
    // Physics
    float theta;        // Barnes-Hut opening angle
    float G;            // Gravitational constant
    float softening;    // Softening length
    
    // Dark matter halo (logarithmic potential)
    // a = -haloVcSq * r / (r^2 + haloCoreSq)
    // Set haloVcSq = 0 to disable
    float haloVcSq;     // v_c^2 (0 = no halo)
    float haloCoreSq;   // r_c^2
    
    // Timestep
    int subSteps;       // physics sub-steps per rendered frame
    float dt;           // timestep per sub-step
    
    // Camera
    float initialZoom;
    float cameraX, cameraY;
    
    // Capacity headroom for spawned bodies
    int extraCapacity;
};

struct Scenario {
    SimulationConfig config;
    std::vector<BodyHost> bodies;
};

// =====================================================================
// Utility: generate a disk galaxy at a given position with bulk velocity
// =====================================================================
inline void addDiskGalaxy(std::vector<BodyHost>& bodies,
                          float centerX, float centerY,
                          float bulkVx, float bulkVy,
                          float centralMass, float centralRadius,
                          int numParticles, float diskRadius,
                          int numArms, float armWind, float armSpread,
                          float particleMass, float particleRadius,
                          float G, float softening,
                          float haloVcSq = 0.0f, float haloCoreSq = 0.0f,
                          bool clockwise = false) {
    
    // Central supermassive body
    bodies.push_back({centerX, centerY, bulkVx, bulkVy, centralMass, centralRadius});
    
    int bulgeCount = numParticles / 5;
    float direction = clockwise ? -1.0f : 1.0f;
    
    // Helper: circular velocity at radius r from THIS galaxy's center
    auto vCirc = [&](float r) -> float {
        float dist = std::sqrt(r * r + softening * softening);
        float v_kep_sq = G * centralMass * r * r / (dist * dist * dist);
        // If halo is centered at (0,0) and galaxy is offset, 
        // halo contribution is per-galaxy local only if haloVcSq > 0
        // For collision scenarios (haloVcSq=0), just Keplerian
        float v_halo_sq = haloVcSq * r * r / (r * r + haloCoreSq);
        return std::sqrt(v_kep_sq + v_halo_sq);
    };
    
    // --- Central Bulge ---
    for (int i = 0; i < bulgeCount; i++) {
        float u1 = std::max(0.0001f, (float)rand() / RAND_MAX);
        float u2 = (float)rand() / RAND_MAX;
        
        float raw_r = std::sqrt(-2.0f * std::log(u1)) * (diskRadius * 0.04f);
        float r = std::max(raw_r, 3.0f);
        float angle = u2 * 2.0f * (float)M_PI;
        
        float v = vCirc(r);
        float vr = ((float)rand() / RAND_MAX - 0.5f) * 0.10f * v;
        
        float px = centerX + std::cos(angle) * r;
        float py = centerY + std::sin(angle) * r;
        float vx = bulkVx + direction * (-std::sin(angle) * v + std::cos(angle) * vr);
        float vy = bulkVy + direction * ( std::cos(angle) * v + std::sin(angle) * vr);
        
        bodies.push_back({px, py, vx, vy, particleMass, particleRadius});
    }
    
    // --- Spiral Arms ---
    for (int i = 0; i < numParticles - bulgeCount; i++) {
        int arm = rand() % numArms;
        float armOffset = arm * (2.0f * (float)M_PI / numArms);
        
        float t = (float)rand() / RAND_MAX;
        float r = (0.03f + t * t * 0.97f) * diskRadius;
        
        float spiralAngle = armOffset + armWind * std::log(r / 30.0f + 1.0f);
        float spread = ((float)rand() / RAND_MAX - 0.5f) * armSpread * std::sqrt(r / 50.0f);
        float angle = spiralAngle + spread;
        
        float px = centerX + std::cos(angle) * r;
        float py = centerY + std::sin(angle) * r;
        
        float orbitR = std::max(std::sqrt((px-centerX)*(px-centerX) + (py-centerY)*(py-centerY)), 1.0f);
        float v = vCirc(orbitR);
        float posAngle = std::atan2(py - centerY, px - centerX);
        
        float vr = ((float)rand() / RAND_MAX - 0.5f) * 0.06f * v;
        
        float vx = bulkVx + direction * (-std::sin(posAngle) * v + std::cos(posAngle) * vr);
        float vy = bulkVy + direction * ( std::cos(posAngle) * v + std::sin(posAngle) * vr);
        
        bodies.push_back({px, py, vx, vy, particleMass, particleRadius});
    }
}
