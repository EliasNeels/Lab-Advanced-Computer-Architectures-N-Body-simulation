#pragma once

#include "../scenario.h"

// =====================================================================
// Scenario 8: Zen Nebula
// A dense, massive cold cloud of 500k particles.
// Designed for ultra-smooth, non-flickering visual flow.
// =====================================================================

inline Scenario createZenNebula() {
    Scenario s;
    srand(888);
    
    s.config.name         = "Zen Nebula (Ultra-Smooth)";
    s.config.theta        = 0.95f;    // Balanced culling
    s.config.G            = 0.0005f;  // Weak gravity for slow evolution
    s.config.softening    = 8.0f;     // High softening for "fluid" feel
    s.config.haloVcSq     = 0.0f;
    s.config.haloCoreSq   = 100.0f;
    s.config.subSteps     = 1;
    s.config.dt           = 0.04f;    // Stable timestep
    
    s.config.camTargetX   = 0.0f;
    s.config.camTargetY   = 0.0f;
    s.config.camTargetZ   = 0.0f;
    s.config.camDistance  = 3000.0f;
    s.config.camTheta     = 0.4f;     
    s.config.camPhi       = 0.6f;
    s.config.extraCapacity = 50000;
    s.config.twinkleAmount = 0.0f;
    
    int numBodies = 50000;
    s.bodies.reserve(numBodies);
    
    // Central anchoring mass (gentle)
    s.bodies.push_back({0, 0, 0, 0, 0, 0, 50000.0f, 10.0f});
    
    for (int i = 1; i < numBodies; i++) {
        // Gaussian-like distribution for a soft cloud
        float u1 = (float)rand() / RAND_MAX;
        float u2 = (float)rand() / RAND_MAX;
        float r = 1200.0f * std::sqrt(-2.0f * std::log(std::max(u1, 0.0001f)));
        
        float theta = acosf(1.0f - 2.0f * ((float)rand() / RAND_MAX));
        float phi = 2.0f * (float)M_PI * ((float)rand() / RAND_MAX);
        
        float px = r * sinf(theta) * cosf(phi);
        float py = r * sinf(theta) * sinf(phi);
        float pz = r * cosf(theta) * 0.4f; // Smashed sphere (oblate)

        // Very slow, dreamy rotation
        float v = 0.5f * std::sqrt(r); 
        float angle = atan2f(py, px);
        float vx = -sinf(angle) * v + ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        float vy =  cosf(angle) * v + ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        float vz = ((float)rand() / RAND_MAX - 0.5f) * 1.5f;

        s.bodies.push_back({
            px, py, pz,
            vx, vy, vz,
            0.05f, 
            0.8f
        });
    }
    
    return s;
}
