#pragma once

#include "../scenario.h"
#include <vector>

// =====================================================================
// Scenario 7: Gravity Sanity Check
// A central mass with concentric rings of particles.
// Each ring has a theoretically calculated circular velocity: v = sqrt(G * M / r)
// If gravity is working correctly, the rings should remain stable circles.
// =====================================================================

inline Scenario createGravityCheck() {
    Scenario s;
    srand(1337); // Deterministic for testing
    
    s.config.name         = "Gravity Sanity Check (Stable Orbits)";
    s.config.theta        = 0.0f;     // FORCE DIRECT SUM (Exact gravity!)
    s.config.G            = 1.0f;     // Use G=1 for simpler math
    s.config.softening    = 0.1f;    // Very low softening for precision
    s.config.haloVcSq     = 0.0f;
    s.config.haloCoreSq   = 1.0f;
    s.config.subSteps     = 4;        // High precision integration
    s.config.dt           = 0.005f;   // Small timestep for stability
    
    s.config.camTargetX   = 0.0f;
    s.config.camTargetY   = 0.0f;
    s.config.camTargetZ   = 0.0f;
    s.config.camDistance  = 400.0f;
    s.config.camTheta     = 0.0f;
    s.config.camPhi       = 1.2f;
    s.config.extraCapacity = 1000;
    s.config.twinkleAmount = 0.0f;

    float centralMass = 10000.0f;
    
    // 1. Central "Sun"
    s.bodies.push_back({
        0.0f, 0.0f, 0.0f, // pos
        0.0f, 0.0f, 0.0f, // vel
        centralMass,      // mass
        10.0f             // radius (visual only)
    });

    // 2. Add 3 concentric rings at radii 50, 100, 150
    std::vector<float> radii = {50.0f, 100.0f, 150.0f};
    int particlesPerRing = 100;

    for (float r : radii) {
        // v = sqrt(G * M / r)
        float vCirc = std::sqrt(s.config.G * centralMass / r);
        
        for (int i = 0; i < particlesPerRing; i++) {
            float angle = (2.0f * (float)M_PI * i) / particlesPerRing;
            
            float px = r * std::cos(angle);
            float py = r * std::sin(angle);
            float pz = 0.0f;

            // Velocity vector is perpendicular to position vector
            float vx = -vCirc * std::sin(angle);
            float vy =  vCirc * std::cos(angle);
            float vz = 0.0f;

            s.bodies.push_back({
                px, py, pz,
                vx, vy, vz,
                0.01f, // Test particles (negligible mass)
                1.0f
            });
        }
    }

    return s;
}
