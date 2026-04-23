#pragma once

#include "../scenario.h"

// =====================================================================
// Scenario 4: The Cartwheel Collision (Presentation Showcase)
// A dense, small galaxy plunges directly through the center of a
// massive spiral galaxy! The impact creates a beautiful expanding 
// ring of stars out of the original disk in 3D space!
// =====================================================================

inline Scenario createCartwheelCollision() {
    Scenario s;
    srand(777); // Predictable awe-inspiring seed
    
    s.config.name         = "Cartwheel Collision Showcase";
    s.config.theta        = 0.6f;     // Excellent balance of speed & precision
    s.config.G            = 0.001f;
    s.config.softening    = 4.5f;
    s.config.haloVcSq     = 0.0f;
    s.config.haloCoreSq   = 1.0f;
    s.config.subSteps     = 2;        // Fast updates for presentation fluidity
    s.config.dt           = 0.12f;
    
    // Position camera far above for the top-down impact view, slightly tilted
    s.config.camTargetX   = 0.0f;
    s.config.camTargetY   = 0.0f;
    s.config.camTargetZ   = 0.0f;
    s.config.camDistance   = 2200.0f;
    s.config.camTheta     = 0.0f;     
    s.config.camPhi       = 0.8f;     // High angle!
    s.config.extraCapacity = 5000;
    s.config.twinkleAmount = 0.0f;
    
    // --- Target Galaxy (The massive spiral disk) ---
    // Sits perfectly flat (0 tilt), large and beautiful.
    addDiskGalaxy(s.bodies,
         0.0f, 0.0f, 0.0f,          // center at origin
         0.0f, -0.05f, 0.0f,        // drifting slightly downwards
         250000.0f, 8.0f,           // massive black hole
         25000, 450.0f,             // huge disk
         5, 2.5f, 0.2f,             // 5 tight beautiful arms 
         0.015f, 0.5f,          
         s.config.G, s.config.softening,
         0.0f, 0.0f,
         false,                       
         0.0f, 0.0f);               // 0 tilt, flat
    
    // --- The Bullet Galaxy (The dense plunger) ---
    // Spawns high directly above the Z-axis, rocketing perfectly downward
    // to punch a hole straight through the target galaxy's center!
    addDiskGalaxy(s.bodies,
         0.0f, 1200.0f, 0.0f,       // Spawns high on Y axis
         0.0f, -1.8f, 0.0f,         // ROCKETING downward
         100000.0f, 6.0f,           // Very dense core
         10000, 150.0f,             // Small, compact disk
         3, 4.0f, 0.3f,         
         0.01f, 0.4f,           
         s.config.G, s.config.softening,
         0.0f, 0.0f,
         true,                        
         1.57f, 0.0f);              // 90° tilt around X (face-first impact!)
    
    return s;
}
