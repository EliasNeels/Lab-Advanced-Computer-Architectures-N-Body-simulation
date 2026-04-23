#pragma once

#include "../scenario.h"

// =====================================================================
// Scenario 6: 2 Million Particle Benchmark
// Replicates the "N=2M" classical BH sphere benchmark from YouTube
// =====================================================================

inline Scenario createTwoMillionBodies() {
    Scenario s;
    srand(42);
    
    s.config.name         = "2M Particle Massive Benchmark";
    s.config.theta        = 1.2f;     // Ultra-aggressive tree culling! (Crucial for 2M)
    s.config.G            = 0.001f;
    s.config.softening    = 4.5f;
    s.config.haloVcSq     = 0.0f;
    s.config.haloCoreSq   = 1.0f;
    s.config.subSteps     = 1;        // Keep it to 1 step per frame!
    s.config.dt           = 0.08f;
    
    // Zoomed out massive view
    s.config.camTargetX   = 0.0f;
    s.config.camTargetY   = 0.0f;
    s.config.camTargetZ   = 0.0f;
    s.config.camDistance  = 8000.0f;
    s.config.camTheta     = 0.5f;     
    s.config.camPhi       = 0.5f;
    s.config.extraCapacity = 10000;
    s.config.twinkleAmount = 0.0f;
    
    // Instead of addDiskGalaxy, we manually pump in 2,000,000 bodies!
    // We will allocate them as a massive sphere rotating gently.
    
    int numBodies = 2000000;
    
    s.bodies.reserve(numBodies);
    
    // Central supermassive black hole to keep it all anchored
    s.bodies.push_back({
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        250000.0f,
        15.0f // Huge!
    });
    
    // The giant cluster
    for (int i = 1; i < numBodies; i++) {
        float r = 5000.0f * cbrtf((float)rand() / RAND_MAX); 
        float theta = acosf(1.0f - 2.0f * ((float)rand() / RAND_MAX));
        float phi = 2.0f * (float)M_PI * ((float)rand() / RAND_MAX);
        
        float px = r * sinf(theta) * cosf(phi);
        float py = r * sinf(theta) * sinf(phi);
        float pz = r * cosf(theta);
        
        // Massive orbiting velocities (bulk rotation around Z-axis)
        float vx = -py * 0.0015f;
        float vy = px * 0.0015f;
        float vz = ((float)rand() / RAND_MAX - 0.5f) * 0.5f;
        
        s.bodies.push_back({
            px, py, pz,
            vx, vy, vz,
            0.1f, // tiny mass tracers
            0.6f
        });
    }
    
    return s;
}
