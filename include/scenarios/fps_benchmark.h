#pragma once

#include "../scenario.h"
#include <cstdlib>

// =====================================================================
// Scenario 3: FPS Benchmark
// Dense uniform spherical cluster, ideal for stressing the octree 
// traversal and force calculation performance.
// =====================================================================

inline Scenario createFpsBenchmark() {
    Scenario s;
    srand(42);
    
    s.config.name         = "FPS Benchmark";
    s.config.theta        = 0.85f;
    s.config.G            = 0.001f;
    s.config.softening    = 4.0f;
    // No DM halo for generic benchmarking to isolate tree overhead
    s.config.haloVcSq     = 0.0f;
    s.config.haloCoreSq   = 0.0f;
    s.config.subSteps     = 1; // Keep substeps low to see raw max FPS
    s.config.dt           = 0.1f;
    s.config.camTargetX   = 0.0f;
    s.config.camTargetY   = 0.0f;
    s.config.camTargetZ   = 0.0f;
    s.config.camDistance   = 1600.0f;
    s.config.camTheta     = 0.5f;     
    s.config.camPhi       = 0.5f;     
    s.config.extraCapacity = 5000;
    
    int N = 150000;
    float maxRadius = 800.0f;
    
    s.bodies.reserve(N);
    
    for (int i = 0; i < N; ++i) {
        // Random point in sphere using rejection sampling or spherical coords
        float r = maxRadius * std::cbrt((float)rand() / RAND_MAX); // Uniform volume
        float phi = 2.0f * 3.14159265f * ((float)rand() / RAND_MAX);
        float costheta = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
        float sintheta = std::sqrt(1.0f - costheta * costheta);
        
        float px = r * sintheta * std::cos(phi);
        float py = r * sintheta * std::sin(phi);
        float pz = r * costheta;
        
        // Small orbital velocity to keep it dynamic but mostly expanding/collapsing
        float dist = std::sqrt(px*px + py*py + pz*pz);
        float vel = std::sqrt(s.config.G * 10000.0f / (dist + s.config.softening));
        
        // Circular-ish orbit around Z axis
        float vx = -py * vel / dist;
        float vy = px * vel / dist;
        float vz = 0.0f;
        
        BodyHost body;
        body.px = px;
        body.py = py;
        body.pz = pz;
        body.vx = vx;
        body.vy = vy;
        body.vz = vz;
        body.mass = 0.05f;
        body.radius = 0.5f;
        
        s.bodies.push_back(body);
    }
    
    return s;
}
