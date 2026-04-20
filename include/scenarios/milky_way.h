#pragma once

#include "../scenario.h"

// =====================================================================
// Scenario 1: Milky Way Galaxy
// A single spiral galaxy with dark matter halo for flat rotation curve
// =====================================================================

inline Scenario createMilkyWay() {
    Scenario s;
    srand(42);
    
    s.config.name         = "Milky Way Galaxy";
    s.config.theta        = 0.5f;
    s.config.G            = 0.001f;
    s.config.softening    = 5.0f;
    s.config.haloVcSq     = 4.0f;     // v_c = 2.0 → flat rotation curve
    s.config.haloCoreSq   = 2500.0f;   // r_c = 50
    s.config.subSteps     = 4;
    s.config.dt           = 0.1f;
    s.config.initialZoom  = 0.001f;
    s.config.cameraX      = 0.0f;
    s.config.cameraY      = 0.0f;
    s.config.extraCapacity = 5000;
    
    int N = 30000;
    
    addDiskGalaxy(s.bodies,
        0.0f, 0.0f,           // center
        0.0f, 0.0f,           // bulk velocity
        200000.0f, 8.0f,      // central mass, radius
        N, 800.0f,            // num particles, disk radius
        4, 3.5f, 0.35f,       // arms, wind, spread
        0.01f, 0.4f,          // particle mass, particle radius
        s.config.G, s.config.softening,
        s.config.haloVcSq, s.config.haloCoreSq);
    
    return s;
}
