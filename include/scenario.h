#pragma once

#include <vector>
#include <string>
#include <cmath>
#include "body.h"

// =====================================================================
// Simulation Scenario System (3D)
// Each scenario defines: physics parameters + initial body configuration
// =====================================================================

struct SimulationConfig {
    std::string name;
    
    // Physics
    float theta;
    float G;
    float softening;
    
    // Dark matter halo (logarithmic potential)
    float haloVcSq;
    float haloCoreSq;
    
    // Timestep
    int subSteps;
    float dt;
    
    // 3D Camera
    float camTargetX, camTargetY, camTargetZ;
    float camDistance;
    float camTheta, camPhi;
    
    // Capacity headroom
    int extraCapacity;
};

struct Scenario {
    SimulationConfig config;
    std::vector<BodyHost> bodies;
};

// =====================================================================
// Utility: generate a 3D disk galaxy at a given position with tilt
// =====================================================================
inline void addDiskGalaxy(std::vector<BodyHost>& bodies,
                          float centerX, float centerY, float centerZ,
                          float bulkVx, float bulkVy, float bulkVz,
                          float centralMass, float centralRadius,
                          int numParticles, float diskRadius,
                          int numArms, float armWind, float armSpread,
                          float particleMass, float particleRadius,
                          float G, float softening,
                          float haloVcSq = 0.0f, float haloCoreSq = 0.0f,
                          bool clockwise = false,
                          float tiltX = 0.0f, float tiltY = 0.0f) {
    
    // Central supermassive body
    bodies.push_back({centerX, centerY, centerZ, bulkVx, bulkVy, bulkVz, centralMass, centralRadius});
    
    int bulgeCount = numParticles / 5;
    float direction = clockwise ? -1.0f : 1.0f;
    
    // Tilt rotation matrix (rotate disk normal from Z-axis)
    float cTx = cosf(tiltX), sTx = sinf(tiltX);
    float cTy = cosf(tiltY), sTy = sinf(tiltY);
    
    // Apply tilt: first rotate around X, then around Y
    auto applyTilt = [&](float lx, float ly, float lz, float& ox, float& oy, float& oz) {
        // Rotate around X
        float ty = ly * cTx - lz * sTx;
        float tz = ly * sTx + lz * cTx;
        // Rotate around Y
        ox = lx * cTy + tz * sTy;
        oy = ty;
        oz = -lx * sTy + tz * cTy;
    };
    
    auto vCirc = [&](float r) -> float {
        float dist = std::sqrt(r * r + softening * softening);
        float v_kep_sq = G * centralMass * r * r / (dist * dist * dist);
        float v_halo_sq = haloVcSq * r * r / (r * r + haloCoreSq);
        return std::sqrt(v_kep_sq + v_halo_sq);
    };
    
    float diskThickness = diskRadius * 0.02f; // Thin disk (2% of radius)
    
    // --- Central Bulge ---
    for (int i = 0; i < bulgeCount; i++) {
        float u1 = std::max(0.0001f, (float)rand() / RAND_MAX);
        float u2 = (float)rand() / RAND_MAX;
        
        float raw_r = std::sqrt(-2.0f * std::log(u1)) * (diskRadius * 0.04f);
        float r = std::max(raw_r, 3.0f);
        float angle = u2 * 2.0f * (float)M_PI;
        float z_offset = ((float)rand() / RAND_MAX - 0.5f) * diskThickness * 2.0f;
        
        float v = vCirc(r);
        float vr = ((float)rand() / RAND_MAX - 0.5f) * 0.10f * v;
        
        // Local disk coordinates (before tilt)
        float lx = std::cos(angle) * r;
        float ly = std::sin(angle) * r;
        float lz = z_offset;
        
        float lvx = direction * (-std::sin(angle) * v + std::cos(angle) * vr);
        float lvy = direction * ( std::cos(angle) * v + std::sin(angle) * vr);
        float lvz = 0.0f;
        
        // Apply tilt rotation
        float px, py, pz, vx, vy, vz;
        applyTilt(lx, ly, lz, px, py, pz);
        applyTilt(lvx, lvy, lvz, vx, vy, vz);
        
        bodies.push_back({
            centerX + px, centerY + py, centerZ + pz,
            bulkVx + vx, bulkVy + vy, bulkVz + vz,
            particleMass, particleRadius
        });
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
        float z_offset = ((float)rand() / RAND_MAX - 0.5f) * diskThickness;
        
        float lx = std::cos(angle) * r;
        float ly = std::sin(angle) * r;
        float lz = z_offset;
        
        float orbitR = std::max(std::sqrt(lx*lx + ly*ly), 1.0f);
        float v = vCirc(orbitR);
        float posAngle = std::atan2(ly, lx);
        
        float vr = ((float)rand() / RAND_MAX - 0.5f) * 0.06f * v;
        
        float lvx = direction * (-std::sin(posAngle) * v + std::cos(posAngle) * vr);
        float lvy = direction * ( std::cos(posAngle) * v + std::sin(posAngle) * vr);
        float lvz = 0.0f;
        
        float px, py, pz, vx, vy, vz;
        applyTilt(lx, ly, lz, px, py, pz);
        applyTilt(lvx, lvy, lvz, vx, vy, vz);
        
        bodies.push_back({
            centerX + px, centerY + py, centerZ + pz,
            bulkVx + vx, bulkVy + vy, bulkVz + vz,
            particleMass, particleRadius
        });
    }
}
