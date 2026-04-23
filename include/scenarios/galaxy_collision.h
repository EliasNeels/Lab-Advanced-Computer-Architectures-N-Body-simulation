#pragma once

#include "../scenario.h"

// =====================================================================
// Scenario 2: Galaxy Collision (3D, Antennae-style)
// Two tilted spiral galaxies on a collision course
// =====================================================================

inline Scenario createGalaxyCollision() {
    Scenario s;
    srand(123);
    
    s.config.name         = "Galaxy Collision";
    s.config.theta        = 0.5f;
    s.config.G            = 0.001f;
    s.config.softening    = 4.0f;
    s.config.haloVcSq     = 0.0f;
    s.config.haloCoreSq   = 1.0f;
    s.config.subSteps     = 4;
    s.config.dt           = 0.08f;
    s.config.camTargetX   = 50.0f;
    s.config.camTargetY   = 0.0f;
    s.config.camTargetZ   = 0.0f;
    s.config.camDistance   = 2000.0f;
    s.config.camTheta     = 0.0f;
    s.config.camPhi       = 0.6f;     // elevated view
    s.config.extraCapacity = 5000;
    s.config.twinkleAmount = 1.0f;
    
    // Galaxy A: Large, tilted 15° around X axis
    addDiskGalaxy(s.bodies,
        -300.0f, 0.0f, 100.0f,      // center position (3D offset)
         0.6f, 0.0f, -0.15f,        // bulk velocity
         150000.0f, 7.0f,
         15000, 350.0f,
         4, 3.0f, 0.3f,
         0.01f, 0.4f,
         s.config.G, s.config.softening,
         0.0f, 0.0f,
         false,                       // CCW
         0.26f, 0.0f);              // tilt ~15° around X
    
    // Galaxy B: Smaller, tilted -30° around Y axis (perpendicular!)
    addDiskGalaxy(s.bodies,
         400.0f, 0.0f, -80.0f,      // center position
        -0.8f, 0.0f, 0.1f,          // bulk velocity
         80000.0f, 5.0f,
         10000, 220.0f,
         3, 4.0f, 0.4f,
         0.01f, 0.35f,
         s.config.G, s.config.softening,
         0.0f, 0.0f,
         true,                        // CW
         0.0f, -0.52f);             // tilt ~-30° around Y
    
    return s;
}
