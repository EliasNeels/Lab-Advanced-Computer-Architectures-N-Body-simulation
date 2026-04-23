#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "body.h"

// Spawn state shared between callback and main loop
struct SpawnState {
    bool justPressed = false;
    bool justReleased = false;
    bool holding = false;
    double pressX = 0, pressY = 0;
    double releaseX = 0, releaseY = 0;
};

class Renderer {
public:
    Renderer(int width, int height);
    ~Renderer();

    bool init();
    void updateVBO(Bodies& bodies, cudaStream_t stream = 0);
    void render(int bodyCount);
    void processInput();
    bool shouldClose() const;
    void swapBuffers();

    void screenToWorld(double screenX, double screenY, float& worldX, float& worldY);

    GLFWwindow* getWindow() const { return window; }
    
    // 3D orbital camera
    void setCamera3D(float targetX, float targetY, float targetZ,
                     float distance, float theta, float phi) {
        camTargetX = targetX; camTargetY = targetY; camTargetZ = targetZ;
        camDistance = distance; camTheta = theta; camPhi = phi;
    }
    
    void setTwinkleAmount(float amt) { twinkleAmount = amt; }
    
    SpawnState spawnState;

private:
    int windowWidth, windowHeight;
    GLFWwindow* window;

    GLuint vao, vbo;
    GLuint shaderProgram;

    float* d_vbo_data;
    float* h_vbo_data;   // Pinned host staging buffer
    int maxBodies;

    // Cached uniform locations (avoid glGetUniformLocation every frame)
    GLint loc_uMVP;
    GLint loc_uScreenHeight;
    GLint loc_uTime;
    GLint loc_uTwinkleAmount;

    // 3D orbital camera (spherical coordinates)
    float camTargetX, camTargetY, camTargetZ;  // look-at point
    float camDistance;                           // distance from target
    float camTheta;                             // horizontal angle (radians)
    float camPhi;                               // vertical angle (radians)
    
    // Mouse state for rotation
    bool mouseRotating;
    double lastMouseX, lastMouseY;
    float twinkleAmount;

    GLuint compileShaders();
    
    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
};
