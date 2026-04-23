#pragma once

#include "../scenario.h"

// =====================================================================
// Scenario 1: Milky Way Galaxy (3D)
// Flat disk with dark matter halo, viewed from a slight angle
// =====================================================================

inline Scenario createMilkyWay() {
    Scenario s;
    srand(42);
    
    s.config.name         = "Milky Way Galaxy";
    s.config.theta        = 0.5f;
    s.config.G            = 0.001f;
    s.config.softening    = 5.0f;
    s.config.haloVcSq     = 4.0f;
    s.config.haloCoreSq   = 2500.0f;
    s.config.subSteps     = 4;
    s.config.dt           = 0.1f;
    s.config.twinkleAmount = 0.0f;
    s.config.camTargetX   = 0.0f;
    s.config.camTargetY   = 0.0f;
    s.config.camTargetZ   = 0.0f;
    s.config.camDistance   = 1500.0f;
    s.config.camTheta     = 0.3f;     // slightly rotated
    s.config.camPhi       = 0.5f;     // looking down at ~30°
    s.config.extraCapacity = 5000;
    
    int N = 30000;
    
    addDiskGalaxy(s.bodies,
        0.0f, 0.0f, 0.0f,         // center (3D)
        0.0f, 0.0f, 0.0f,         // bulk velocity
        200000.0f, 8.0f,           // central mass, radius
        N, 800.0f,                 // particles, disk radius
        4, 3.5f, 0.35f,           // arms, wind, spread
        0.01f, 0.4f,              // particle mass, radius
        s.config.G, s.config.softening,
        s.config.haloVcSq, s.config.haloCoreSq,
        false,                     // counter-clockwise
        0.0f, 0.0f);              // no tilt (flat in XY plane)
    
    return s;
}
