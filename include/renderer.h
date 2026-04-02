#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "body.h"

// Spawn state shared between callback and main loop
struct SpawnState {
    bool justPressed = false;   // Set by callback on press
    bool justReleased = false;  // Set by callback on release
    bool holding = false;       // True while button is held
    double pressX = 0, pressY = 0;     // Screen coords where pressed
    double releaseX = 0, releaseY = 0; // Screen coords where released
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
    float getCameraX() const { return cameraX; }
    float getCameraY() const { return cameraY; }
    float getCameraZoom() const { return cameraZoom; }
    
    SpawnState spawnState;

private:
    int windowWidth, windowHeight;
    GLFWwindow* window;

    GLuint vao, vbo;
    GLuint shaderProgram;

    float* d_vbo_data;
    float* h_vbo_data;
    int maxBodies;

    float cameraX, cameraY, cameraZoom;

    GLuint compileShaders();
    
    // GLFW callback (static because GLFW is C-style)
    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
};
