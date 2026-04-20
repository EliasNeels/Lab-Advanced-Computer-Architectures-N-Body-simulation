#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <string>

#include "../include/body.h"
#include "../include/tree.h"
#include "../include/BoundingBox.h"
#include "../include/cuda_utils.h"
#include "../include/renderer.h"
#include "../include/kernels.cuh"
#include "../include/scenario.h"

// ===== Scenario headers =====
#include "../include/scenarios/milky_way.h"
#include "../include/scenarios/galaxy_collision.h"

int main(int argc, char** argv) {
    // =====================================================================
    // Scenario Selection
    // =====================================================================
    std::cout << std::endl;
    std::cout << "╔══════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║     CUDA N-Body Simulation  ★ 3D ★             ║" << std::endl;
    std::cout << "║     Barnes-Hut Octree | Leapfrog Integration   ║" << std::endl;
    std::cout << "╠══════════════════════════════════════════════════╣" << std::endl;
    std::cout << "║  Select simulation:                             ║" << std::endl;
    std::cout << "║    1. Milky Way Galaxy                          ║" << std::endl;
    std::cout << "║    2. Galaxy Collision  ★                       ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════╝" << std::endl;
    
    int selection = 0;
    
    if (argc > 1) {
        selection = std::atoi(argv[1]);
    }
    
    if (selection < 1 || selection > 2) {
        std::cout << "\n  Enter choice (1-2): ";
        std::cin >> selection;
    }
    
    if (selection < 1 || selection > 2) {
        std::cerr << "Invalid selection!" << std::endl;
        return -1;
    }
    
    // =====================================================================
    // Load Scenario
    // =====================================================================
    Scenario scenario;
    switch (selection) {
        case 1: scenario = createMilkyWay();         break;
        case 2: scenario = createGalaxyCollision();   break;
    }
    
    const SimulationConfig& cfg = scenario.config;
    int N = (int)scenario.bodies.size();
    int maxCapacity = N + cfg.extraCapacity;
    
    std::cout << "\n  ► Loading: " << cfg.name << " (3D)" << std::endl;
    std::cout << "  ► Bodies: " << N << "  |  Max: " << maxCapacity << std::endl;
    std::cout << "  ► θ=" << cfg.theta << "  G=" << cfg.G 
              << "  ε=" << cfg.softening << std::endl;
    if (cfg.haloVcSq > 0) {
        std::cout << "  ► DM Halo: v_c=" << std::sqrt(cfg.haloVcSq)
                  << "  r_c=" << std::sqrt(cfg.haloCoreSq) << std::endl;
    }
    std::cout << std::endl;
    
    // =====================================================================
    // GPU Initialization
    // =====================================================================
    Bodies d_bodies;
    bodiesAllocDevice(d_bodies, maxCapacity);
    bodiesUpload(d_bodies, scenario.bodies.data(), N);
    
    scenario.bodies.clear();
    scenario.bodies.shrink_to_fit();

    BoundingBox* d_bbox = nullptr;
    CUDA_CHECK(cudaMalloc(&d_bbox, sizeof(BoundingBox)));
    cudaDeviceSynchronize();

    std::cout << "[1/5] Running bounding box reduction..." << std::endl;
    launchBoundingBox(d_bodies, d_bbox, 0);
    
    uint32_t* d_mortonKeys = nullptr;
    uint32_t* d_mortonKeysOut = nullptr;
    int* d_indicesIn = nullptr;
    int* d_sortedIndices = nullptr;
    void* d_tempStorage = nullptr;
    size_t tempStorageBytes = 0;

    CUDA_CHECK(cudaMalloc(&d_mortonKeys,    maxCapacity * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_mortonKeysOut, maxCapacity * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_indicesIn,     maxCapacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sortedIndices, maxCapacity * sizeof(int)));

    Bodies d_scratch;
    bodiesAllocDevice(d_scratch, maxCapacity);

    std::cout << "[2/5] Running Morton Sort..." << std::endl;
    launchMortonSort(d_bodies, d_scratch, d_bbox, d_mortonKeys, d_mortonKeysOut, 
                     d_indicesIn, d_sortedIndices, d_tempStorage, tempStorageBytes, 0);

    Octree tree;
    tree.nodes = nullptr;
    tree.parents = nullptr;
    tree.nodeCount = nullptr;
    tree.maxNodes = 0;
    int leafCapacity = 16;
    
    Renderer renderer(1600, 1000);
    if (!renderer.init()) {
        std::cerr << "Failed to initialize renderer!" << std::endl;
        return -1;
    }
    renderer.setCamera3D(cfg.camTargetX, cfg.camTargetY, cfg.camTargetZ,
                         cfg.camDistance, cfg.camTheta, cfg.camPhi);
    
    auto lastTime = std::chrono::high_resolution_clock::now();
    int frames = 0;
    
    auto spawnStartTime = std::chrono::high_resolution_clock::now();
    float spawnWorldX = 0, spawnWorldY = 0;
    float massGrowthRate = 500.0f;

    std::cout << "[3/5] Initializing forces for Leapfrog..." << std::endl;
    launchBoundingBox(d_bodies, d_bbox, 0);
    launchMortonSort(d_bodies, d_scratch, d_bbox, d_mortonKeys, d_mortonKeysOut, 
                     d_indicesIn, d_sortedIndices, d_tempStorage, tempStorageBytes, 0);
    launchBuildTree(tree, d_bodies, d_bbox, d_mortonKeysOut, leafCapacity, 0);
    launchForceCalculation(d_bodies, tree, cfg.theta, cfg.G, cfg.softening, 
                           cfg.haloVcSq, cfg.haloCoreSq, 0);
    cudaDeviceSynchronize();

    std::cout << "[4/5] Setup complete!" << std::endl;
    std::cout << "[5/5] Starting 3D Simulation..." << std::endl;
    std::cout << std::endl;
    std::cout << "╔══════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  3D Controls:                                   ║" << std::endl;
    std::cout << "║    RIGHT-CLICK + DRAG  — Orbit camera           ║" << std::endl;
    std::cout << "║    Scroll              — Zoom in/out            ║" << std::endl;
    std::cout << "║    WASD                — Pan camera target      ║" << std::endl;
    std::cout << "║    SPACE / SHIFT       — Move up / down         ║" << std::endl;
    std::cout << "║    Arrow Up/Dn         — Zoom in/out (keys)     ║" << std::endl;
    std::cout << "║    R                   — Reset camera            ║" << std::endl;
    std::cout << "║    LEFT-CLICK          — Spawn body             ║" << std::endl;
    std::cout << "║    ESC                 — Quit                    ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════╝" << std::endl;
    
    // =====================================================================
    // Main Simulation Loop
    // =====================================================================
    while (!renderer.shouldClose()) {
        // ===== Handle mouse spawning =====
        SpawnState& ss = renderer.spawnState;
        GLFWwindow* win = renderer.getWindow();
        
        if (ss.justPressed) {
            ss.justPressed = false;
            renderer.screenToWorld(ss.pressX, ss.pressY, spawnWorldX, spawnWorldY);
            spawnStartTime = std::chrono::high_resolution_clock::now();
        }
        
        bool released = ss.justReleased;
        if (!released && ss.holding && glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE) {
            released = true;
            ss.holding = false;
            double mx, my;
            glfwGetCursorPos(win, &mx, &my);
            ss.releaseX = mx;
            ss.releaseY = my;
        }
        
        if (released) {
            ss.justReleased = false;
            
            auto now = std::chrono::high_resolution_clock::now();
            float holdSeconds = std::chrono::duration<float>(now - spawnStartTime).count();
            float spawnMass = 10.0f + holdSeconds * massGrowthRate;
            float spawnRadius = std::cbrt(spawnMass) * 0.5f;
            
            float releaseWorldX, releaseWorldY;
            renderer.screenToWorld(ss.releaseX, ss.releaseY, releaseWorldX, releaseWorldY);
            
            float velScale = 0.5f;
            float vx = (releaseWorldX - spawnWorldX) * velScale;
            float vy = (releaseWorldY - spawnWorldY) * velScale;
            
            if (d_bodies.count < d_bodies.capacity) {
                BodyHost newBody;
                newBody.px = spawnWorldX;
                newBody.py = spawnWorldY;
                newBody.pz = 0.0f;  // Spawn at z=0
                newBody.vx = vx;
                newBody.vy = vy;
                newBody.vz = 0.0f;
                newBody.mass = spawnMass;
                newBody.radius = spawnRadius;
                bodiesUpload(d_bodies, &newBody, 1);
                
                std::cout << "★ Spawned body! Mass: " << spawnMass 
                         << " Total: " << d_bodies.count << std::endl;
            }
        }

        // ===== Physics sub-stepping =====
        for (int step = 0; step < cfg.subSteps; step++) {
            launchLeapfrogKickDrift(d_bodies, cfg.dt, 0);
            
            launchBoundingBox(d_bodies, d_bbox, 0);
            
            if (step % 2 == 0) {
                launchMortonSort(d_bodies, d_scratch, d_bbox, d_mortonKeys, d_mortonKeysOut, 
                                 d_indicesIn, d_sortedIndices, d_tempStorage, tempStorageBytes, 0);
            }
            
            launchBuildTree(tree, d_bodies, d_bbox, d_mortonKeysOut, leafCapacity, 0);
            launchForceCalculation(d_bodies, tree, cfg.theta, cfg.G, cfg.softening, 
                                   cfg.haloVcSq, cfg.haloCoreSq, 0);
            
            launchLeapfrogKick(d_bodies, cfg.dt, 0);
        }

        // ===== Render =====
        renderer.updateVBO(d_bodies, 0);
        cudaDeviceSynchronize();
        
        renderer.render(d_bodies.count);
        renderer.swapBuffers();
        
        frames++;
        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - lastTime).count() >= 1) {
            std::string title = "✦ " + cfg.name + " [3D] | FPS: " + std::to_string(frames) 
                              + " | Bodies: " + std::to_string(d_bodies.count);
            glfwSetWindowTitle(renderer.getWindow(), title.c_str());
            frames = 0;
            lastTime = now;
        }
    }
    
    std::cout << std::endl << "Window closed. Cleaning up..." << std::endl;
    
    cudaFree(d_bbox);
    cudaFree(d_mortonKeys);
    cudaFree(d_mortonKeysOut);
    cudaFree(d_indicesIn);
    cudaFree(d_sortedIndices);
    if (d_tempStorage) cudaFree(d_tempStorage);
    if(tree.nodes) cudaFree(tree.nodes);
    if(tree.parents) cudaFree(tree.parents);
    if(tree.nodeCount) cudaFree(tree.nodeCount);
    bodiesFreeDevice(d_bodies);
    bodiesFreeDevice(d_scratch);

    return 0;
}