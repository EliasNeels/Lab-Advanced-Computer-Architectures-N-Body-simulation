#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <string>

#include "../include/body.h"
#include "../include/quadtree.h"
#include "../include/cuda_utils.h"
#include "../include/renderer.h"

// Declaration from bounding_box.cu
struct BoundingBox {
    float min_x, min_y;
    float max_x, max_y;
};
void launchBoundingBox(const Bodies& bodies, BoundingBox* d_bbox, cudaStream_t stream = 0);

// Declaration from morton_sort.cu
void launchMortonSort(Bodies& bodies, const BoundingBox* d_bbox,
                      uint32_t* d_mortonKeys, int* d_sortedIndices,
                      void* d_tempStorage, size_t& tempStorageBytes,
                      cudaStream_t stream = 0);

// Declaration from build_tree.cu
void launchBuildTree(Quadtree& tree, Bodies& bodies, const BoundingBox* d_bbox, const uint32_t* d_mortonCodes,
                     int leafCapacity, cudaStream_t stream = 0);

int main() {
    std::cout << "Starting CUDA N-Body Simulation..." << std::endl;

    // 1. Initialize bodies
    int N = 50000;
    int maxCapacity = N + 5000;
    Bodies d_bodies;
    bodiesAllocDevice(d_bodies, maxCapacity);
    
    std::vector<BodyHost> h_bodies(N);
    srand(42);
    
    // ===== Milky Way-style Spiral Galaxy =====
    float centralMass = 200000.0f;
    float galaxyRadius = 800.0f;
    int numArms = 4;
    float armSpread = 0.4f;
    float armWindFactor = 4.0f;
    int bulgeCount = N / 5;
    
    // Central supermassive black hole
    h_bodies[0] = {0, 0, 0, 0, centralMass, 8.0f};
    
    // --- Central Bulge (dense core) ---
    for (int i = 1; i <= bulgeCount; i++) {
        float u1 = std::max(0.0001f, (float)rand() / RAND_MAX);
        float u2 = (float)rand() / RAND_MAX;
        float r = std::sqrt(-2.0f * std::log(u1)) * (galaxyRadius * 0.08f);
        float angle = u2 * 2.0f * (float)M_PI;
        
        float orbitR = std::max(r, 3.0f);
        float v = std::sqrt(0.001f * centralMass / orbitR);
        h_bodies[i] = {
            std::cos(angle) * r, std::sin(angle) * r,
            -std::sin(angle) * v, std::cos(angle) * v,
            1.0f, 0.5f
        };
    }
    
    // --- Spiral Arms ---
    for (int i = bulgeCount + 1; i < N; i++) {
        int arm = rand() % numArms;
        float armOffset = arm * (2.0f * (float)M_PI / numArms);
        
        float t = (float)rand() / RAND_MAX;
        float r = (0.1f + t * 0.9f) * galaxyRadius;
        
        float spiralAngle = armOffset + armWindFactor * std::log(r / 50.0f + 1.0f);
        float spread = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * armSpread * (0.3f + 0.7f * t);
        float angle = spiralAngle + spread;
        float jitter = ((float)rand() / RAND_MAX - 0.5f) * 30.0f;
        float finalR = r + jitter;
        
        float px = std::cos(angle) * finalR;
        float py = std::sin(angle) * finalR;
        float orbitR = std::max(std::sqrt(px*px + py*py), 5.0f);
        float v = std::sqrt(0.001f * centralMass / orbitR);
        float posAngle = std::atan2(py, px);
        
        h_bodies[i] = {
            px, py,
            -std::sin(posAngle) * v, std::cos(posAngle) * v,
            1.0f, 0.4f
        };
    }
    
    bodiesUpload(d_bodies, h_bodies.data(), N);

    BoundingBox* d_bbox = nullptr;
    CUDA_CHECK(cudaMalloc(&d_bbox, sizeof(BoundingBox)));
    cudaDeviceSynchronize();

    std::cout << "Running bounding box reduction..." << std::endl;
    launchBoundingBox(d_bodies, d_bbox, 0);
    
    uint32_t* d_mortonKeys = nullptr;
    int* d_sortedIndices = nullptr;
    void* d_tempStorage = nullptr;
    size_t tempStorageBytes = 0;

    CUDA_CHECK(cudaMalloc(&d_mortonKeys, maxCapacity * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_sortedIndices, maxCapacity * sizeof(int)));

    std::cout << "Running Morton Sort..." << std::endl;
    launchMortonSort(d_bodies, d_bbox, d_mortonKeys, d_sortedIndices, d_tempStorage, tempStorageBytes, 0);

    // Build Tree
    Quadtree tree;
    tree.nodes = nullptr;
    tree.parents = nullptr;
    tree.nodeCount = nullptr;
    tree.maxNodes = 0;
    int leafCapacity = 16;

    // Physics parameters
    float theta = 0.5f;
    float G = 0.001f;
    float softening = 1.0f;
    float dt = 0.05f;

    // Forward declarations
    void launchForceCalculation(Bodies& bodies, const Quadtree& tree, float theta, float G, float softening, cudaStream_t stream = 0);
    void launchLeapfrogKickDrift(Bodies& bodies, float dt, cudaStream_t stream = 0);
    void launchLeapfrogKick(Bodies& bodies, float dt, cudaStream_t stream = 0);
    void launchCollisionDetection(Bodies& bodies, const Quadtree& tree, cudaStream_t stream = 0);
    
    // Renderer setup
    Renderer renderer(1200, 900);
    if (!renderer.init()) {
        std::cerr << "Failed to initialize renderer!" << std::endl;
        return -1;
    }
    
    auto lastTime = std::chrono::high_resolution_clock::now();
    int frames = 0;
    
    // ===== Spawning state =====
    auto spawnStartTime = std::chrono::high_resolution_clock::now();
    float spawnWorldX = 0, spawnWorldY = 0;
    float massGrowthRate = 500.0f; // Mass units per second of holding

    std::cout << "Initializing forces for Leapfrog..." << std::endl;
    launchBoundingBox(d_bodies, d_bbox, 0);
    launchMortonSort(d_bodies, d_bbox, d_mortonKeys, d_sortedIndices, d_tempStorage, tempStorageBytes, 0);
    launchBuildTree(tree, d_bodies, d_bbox, d_mortonKeys, leafCapacity, 0);
    launchForceCalculation(d_bodies, tree, theta, G, softening, 0);
    cudaDeviceSynchronize();

    std::cout << "Starting Graphic Simulation Loop..." << std::endl;
    std::cout << "Controls: WASD=pan, ArrowUp/Down=zoom, LEFT-CLICK=spawn body (hold to grow mass, drag to aim, release to shoot)" << std::endl;
    
    while (!renderer.shouldClose()) {
        // ===== Handle mouse spawning =====
        SpawnState& ss = renderer.spawnState;
        GLFWwindow* win = renderer.getWindow();
        
        // Detect press via callback (reliable)
        if (ss.justPressed) {
            ss.justPressed = false;
            renderer.screenToWorld(ss.pressX, ss.pressY, spawnWorldX, spawnWorldY);
            spawnStartTime = std::chrono::high_resolution_clock::now();
            std::cout << "Charging body at (" << spawnWorldX << ", " << spawnWorldY << ")..." << std::endl;
        }
        
        // Detect release via callback OR polling fallback (touchpad fix)
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
                newBody.vx = vx;
                newBody.vy = vy;
                newBody.mass = spawnMass;
                newBody.radius = spawnRadius;
                bodiesUpload(d_bodies, &newBody, 1);
                
                std::cout << "Spawned body! Mass: " << spawnMass 
                         << " Vel: (" << vx << ", " << vy << ")"
                         << " Total bodies: " << d_bodies.count << std::endl;
            }
        }

        // Leapfrog Phase 1 & 2: Kick (half step) + Drift (full step)
        launchLeapfrogKickDrift(d_bodies, dt, 0);
        
        // Recompute forces for Phase 3 (needed for the next Kick)
        launchBoundingBox(d_bodies, d_bbox, 0);
        launchMortonSort(d_bodies, d_bbox, d_mortonKeys, d_sortedIndices, d_tempStorage, tempStorageBytes, 0);
        launchBuildTree(tree, d_bodies, d_bbox, d_mortonKeys, leafCapacity, 0);
        launchForceCalculation(d_bodies, tree, theta, G, softening, 0);
        
        // Leapfrog Phase 4: Kick (half step) with new forces
        launchLeapfrogKick(d_bodies, dt, 0);

        // Optional: Collision handling (best done after position update)
        launchCollisionDetection(d_bodies, tree, 0);
        
        // Render
        renderer.updateVBO(d_bodies, 0);
        cudaDeviceSynchronize();
        
        renderer.render(d_bodies.count);
        renderer.swapBuffers();
        
        // FPS counter
        frames++;
        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - lastTime).count() >= 1) {
            std::string title = "CUDA N-Body Barnes-Hut | FPS: " + std::to_string(frames) + " | Bodies: " + std::to_string(d_bodies.count);
            glfwSetWindowTitle(renderer.getWindow(), title.c_str());
            frames = 0;
            lastTime = now;
        }
    }
    
    std::cout << "Window closed. Cleaning up..." << std::endl;
    
    // Cleanup
    cudaFree(d_bbox);
    cudaFree(d_mortonKeys);
    cudaFree(d_sortedIndices);
    if (d_tempStorage) cudaFree(d_tempStorage);
    if(tree.nodes) cudaFree(tree.nodes);
    if(tree.parents) cudaFree(tree.parents);
    if(tree.nodeCount) cudaFree(tree.nodeCount);
    bodiesFreeDevice(d_bodies);

    return 0;
}