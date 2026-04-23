#pragma once

#include "../scenario.h"

// =====================================================================
// Scenario 5: Tri-Galaxy Cluster Chaos (Presentation Showcase)
// Three massive spiral galaxies spawned extremely close to each other,
// immediately overlapping and ripping each other into chaotic, stunning
// star streams and Lissajous figures!
// =====================================================================

inline Scenario createChaoticAttractors() {
    Scenario s;
    srand(12345); // Chosen for beautiful arm alignments
    
    s.config.name         = "Tri-Galaxy Maelstrom";
    s.config.theta        = 0.65f;    // Fast evaluating
    s.config.G            = 0.001f;
    s.config.softening    = 3.5f;
    s.config.haloVcSq     = 0.0f;
    s.config.haloCoreSq   = 1.0f;
    s.config.subSteps     = 2;
    s.config.dt           = 0.14f;
    
    // Zoomed close enough to see the intense overlapping cores
    s.config.camTargetX   = 0.0f;
    s.config.camTargetY   = 0.0f;
    s.config.camTargetZ   = 0.0f;
    s.config.camDistance  = 1800.0f;
    s.config.camTheta     = 0.4f;     
    s.config.camPhi       = 0.5f;
    s.config.extraCapacity = 5000;
    s.config.twinkleAmount = 0.0f;
    
    // By spawning 3 large disk galaxies extremely tight to the center (radius 300, distance 350)
    // They are instantly entangled and ripping each other's 15,000 stars into deep star streams.
    
    float vSpd = 1.3f;
    
    // Galaxy 1: Right (Plunging Left)
    addDiskGalaxy(s.bodies,
         350.0f, 0.0f, 0.0f,       
         -vSpd, 0.1f, 0.1f,      
         250000.0f, 8.0f,
         14000, 300.0f,
         4, 2.5f, 0.3f,         
         0.015f, 0.4f,           
         s.config.G, s.config.softening,
         0.0f, 0.0f, 
         false, 0.3f, 0.2f);
         
    // Galaxy 2: Top Left (Plunging Bottom-Right)
    addDiskGalaxy(s.bodies,
         -175.0f, 303.0f, 0.0f,       
         vSpd * 0.5f, -vSpd * 0.866f, -0.2f,      
         250000.0f, 8.0f,
         14000, 300.0f,
         5, 3.0f, 0.3f,         
         0.015f, 0.4f,           
         s.config.G, s.config.softening,
         0.0f, 0.0f, 
         true, -0.4f, 0.1f);
         
    // Galaxy 3: Bottom Left (Plunging Top-Right)
    addDiskGalaxy(s.bodies,
         -175.0f, -303.0f, 0.0f,       
         vSpd * 0.5f, vSpd * 0.866f, 0.1f,      
         250000.0f, 8.0f,
         14000, 300.0f,
         3, 2.0f, 0.3f,         
         0.015f, 0.4f,           
         s.config.G, s.config.softening,
         0.0f, 0.0f, 
         false, 0.2f, -0.3f);
    
    return s;
}
