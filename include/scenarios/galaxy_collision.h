#pragma once

#include "../scenario.h"

// =====================================================================
// Scenario 2: Galaxy Collision (Antennae-style)
// Two spiral galaxies on a collision course, creating spectacular
// tidal tails, bridges, and shells as they merge.
// Inspired by NGC 4038/4039 (The Antennae Galaxies)
// =====================================================================

inline Scenario createGalaxyCollision() {
    Scenario s;
    srand(123);
    
    s.config.name         = "Galaxy Collision";
    s.config.theta        = 0.5f;
    s.config.G            = 0.001f;
    s.config.softening    = 4.0f;
    s.config.haloVcSq     = 0.0f;      // No global DM halo — pure N-body collision
    s.config.haloCoreSq   = 1.0f;      // (unused when haloVcSq = 0)
    s.config.subSteps     = 4;
    s.config.dt           = 0.08f;
    s.config.initialZoom  = 0.0007f;   // Wider view to see both galaxies
    s.config.cameraX      = 50.0f;
    s.config.cameraY      = 0.0f;
    s.config.extraCapacity = 5000;
    
    // ===== Galaxy A: Large spiral (Milky Way-like) =====
    // Positioned left, moving right with slight upward drift
    addDiskGalaxy(s.bodies,
        -300.0f, 100.0f,       // center position
         0.6f, -0.15f,         // bulk velocity (approaching)
         150000.0f, 7.0f,      // central mass, radius
         15000, 350.0f,        // num particles, disk radius
         4, 3.0f, 0.3f,        // 4 arms, moderate wind
         0.01f, 0.4f,          // particle mass, radius
         s.config.G, s.config.softening,
         0.0f, 0.0f,           // no per-galaxy halo
         false);               // counter-clockwise
    
    // ===== Galaxy B: Smaller spiral (companion) =====
    // Positioned right, moving left — on a grazing collision course
    addDiskGalaxy(s.bodies,
         400.0f, -80.0f,       // center position
        -0.8f,  0.1f,          // bulk velocity (approaching faster)
         80000.0f, 5.0f,       // smaller central mass
         10000, 220.0f,        // fewer particles, smaller disk
         3, 4.0f, 0.4f,        // 3 arms, tighter wind
         0.01f, 0.35f,         // particle mass, slightly smaller radius
         s.config.G, s.config.softening,
         0.0f, 0.0f,           // no per-galaxy halo
         true);                // clockwise (opposite rotation!)
    
    return s;
}
