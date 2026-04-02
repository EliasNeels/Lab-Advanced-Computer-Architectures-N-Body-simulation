#include "../include/renderer.h"
#include <iostream>
#include <stdexcept>
#include <vector>

// Forward declaration of the CUDA kernel launcher
extern "C" void launchPackVBO(const Bodies& bodies, float* d_vbo, cudaStream_t stream);

// Vertex Shader: positions the points and sets their size based on radius and camera zoom
const char* vertexShaderSource = R"(
#version 450 core
layout (location = 0) in vec4 aData; // x, y, radius, padding

uniform vec2 cameraPos;
uniform float cameraZoom;
uniform vec2 resolution;

out float vRadius;

void main() {
    // Transform from world space to NDC (Normalized Device Coordinates)
    vec2 pos = (aData.xy - cameraPos) * cameraZoom;
    
    // Adjust for aspect ratio
    float aspect = resolution.x / resolution.y;
    pos.x /= aspect;

    gl_Position = vec4(pos, 0.0, 1.0);
    
    // Scale point size based on body radius and camera zoom
    float pointSize = aData.z * cameraZoom * resolution.y * 0.5;
    
    // Clamp point size to visible bounds (driver dependent, usually 1 to 64)
    gl_PointSize = clamp(pointSize, 2.0, 128.0);
    
    vRadius = aData.z;
}
)";

// Fragment Shader: draws a glowing circle with mass-based coloring
const char* fragmentShaderSource = R"(
#version 450 core
out vec4 FragColor;

in float vRadius;

void main() {
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    float distSq = dot(coord, coord);
    
    if (distSq > 1.0) discard;
    
    float alpha = 1.0 - smoothstep(0.3, 1.0, distSq);
    
    // Color by radius/mass:
    // Galaxy bodies (radius ~0.5) → warm white/orange
    // Spawned bodies (radius > 1.0) → bright electric blue
    float t = clamp((vRadius - 0.5) / 2.0, 0.0, 1.0);
    vec3 galaxyColor = vec3(1.0, 0.85, 0.7);  // warm white
    vec3 spawnColor  = vec3(0.2, 0.5, 1.0);    // electric blue
    vec3 color = mix(galaxyColor, spawnColor, t);
    
    // Add a bright core glow
    float core = exp(-distSq * 4.0);
    color += vec3(core * 0.5);
    
    FragColor = vec4(color, alpha);
}
)";

Renderer::Renderer(int width, int height)
    : windowWidth(width), windowHeight(height), window(nullptr), 
      vao(0), vbo(0), shaderProgram(0), d_vbo_data(nullptr), h_vbo_data(nullptr),
      maxBodies(1000000), cameraX(0), cameraY(0), cameraZoom(0.001f) 
{}

Renderer::~Renderer() {
    if (d_vbo_data) cudaFree(d_vbo_data);
    if (h_vbo_data) delete[] h_vbo_data;
    if (vbo) glDeleteBuffers(1, &vbo);
    if (vao) glDeleteVertexArrays(1, &vao);
    if (shaderProgram) glDeleteProgram(shaderProgram);
    if (window) glfwDestroyWindow(window);
    glfwTerminate();
}

bool Renderer::init() {
    // 1. Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return false;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(windowWidth, windowHeight, "CUDA N-Body Barnes-Hut", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create window" << std::endl;
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(0); // Disable VSync for testing max speed

    // 2. Initialize GLEW
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return false;
    }

    // 3. Compile Shaders
    shaderProgram = compileShaders();

    // 4. Setup OpenGL Buffers (VAO/VBO)
    // We allocate a maximum size buffer equivalent to 1,000,000 bodies (vec4 each)
    cudaMalloc(&d_vbo_data, maxBodies * sizeof(float) * 4);
    h_vbo_data = new float[maxBodies * 4];
    
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, maxBodies * sizeof(float) * 4, nullptr, GL_DYNAMIC_DRAW);

    // vertex attribute 0 (aData: vec4)
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // OpenGL State
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Register mouse callback and store 'this' pointer for access from the static callback
    glfwSetWindowUserPointer(window, this);
    glfwSetMouseButtonCallback(window, Renderer::mouseButtonCallback);

    return true;
}

void Renderer::updateVBO(Bodies& bodies, cudaStream_t stream) {
    if (bodies.count == 0) return;

    // Launch kernel to tightly pack X, Y, and Radius from SoA layout to AoS VBO layout
    launchPackVBO(bodies, d_vbo_data, stream);
    
    // Copy to host
    cudaMemcpyAsync(h_vbo_data, d_vbo_data, bodies.count * sizeof(float) * 4, cudaMemcpyDeviceToHost, stream);
    
    // Unmap the VBO before OpenGL tries to draw it
    cudaStreamSynchronize(stream);
    
    // Bind and specify sub-data
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, bodies.count * sizeof(float) * 4, h_vbo_data);
}

void Renderer::render(int bodyCount) {
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    glViewport(0, 0, width, height);

    glClearColor(0.01f, 0.01f, 0.02f, 1.0f); // Deep space blue/black
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(shaderProgram);

    // Process basic camera input (to be replaced by ImGui soon)
    processInput();

    // Set Uniforms
    glUniform2f(glGetUniformLocation(shaderProgram, "cameraPos"), cameraX, cameraY);
    glUniform1f(glGetUniformLocation(shaderProgram, "cameraZoom"), cameraZoom);
    glUniform2f(glGetUniformLocation(shaderProgram, "resolution"), (float)width, (float)height);

    // Draw
    glBindVertexArray(vao);
    glDrawArrays(GL_POINTS, 0, bodyCount);
}

void Renderer::swapBuffers() {
    glfwSwapBuffers(window);
    glfwPollEvents();
}

bool Renderer::shouldClose() const {
    return glfwWindowShouldClose(window);
}

void Renderer::processInput() {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
    
    float speed = 0.5f / cameraZoom;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) cameraY += speed * 0.016f;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) cameraY -= speed * 0.016f;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) cameraX -= speed * 0.016f;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) cameraX += speed * 0.016f;

    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) cameraZoom *= 1.02f;
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) cameraZoom /= 1.02f;
}

GLuint Renderer::compileShaders() {
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return program;
}

void Renderer::screenToWorld(double screenX, double screenY, float& worldX, float& worldY) {
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    float aspect = (float)width / (float)height;
    
    float ndcX = (2.0f * (float)screenX / width - 1.0f) * aspect;
    float ndcY = -(2.0f * (float)screenY / height - 1.0f);
    
    worldX = ndcX / cameraZoom + cameraX;
    worldY = ndcY / cameraZoom + cameraY;
}

void Renderer::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    if (button != GLFW_MOUSE_BUTTON_LEFT) return;
    
    Renderer* self = static_cast<Renderer*>(glfwGetWindowUserPointer(window));
    if (!self) return;
    
    double mx, my;
    glfwGetCursorPos(window, &mx, &my);
    
    if (action == GLFW_PRESS) {
        self->spawnState.justPressed = true;
        self->spawnState.holding = true;
        self->spawnState.pressX = mx;
        self->spawnState.pressY = my;
    } else if (action == GLFW_RELEASE) {
        self->spawnState.justReleased = true;
        self->spawnState.holding = false;
        self->spawnState.releaseX = mx;
        self->spawnState.releaseY = my;
    }
}
