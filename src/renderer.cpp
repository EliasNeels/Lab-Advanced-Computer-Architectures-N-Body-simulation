#include "../include/renderer.h"
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cmath>

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
out vec2 vWorldPos;

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
    gl_PointSize = clamp(pointSize, 1.5, 128.0);
    
    vRadius = aData.z;
    vWorldPos = aData.xy;
}
)";

// Fragment Shader: draws a glowing circle with mass-based coloring
// Uses a multi-layer glow technique for a stunning nebula/galaxy look
const char* fragmentShaderSource = R"(
#version 450 core
out vec4 FragColor;

in float vRadius;
in vec2 vWorldPos;

uniform float uTime;

void main() {
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    float distSq = dot(coord, coord);
    
    if (distSq > 1.0) discard;
    
    float dist = sqrt(distSq);
    
    // ---- Classify body by radius/mass ----
    // Central black hole: radius >= 5.0
    // Bulge stars: radius ~0.5
    // Spiral arm stars: radius ~0.4
    // Spawned bodies: radius > 1.0 (mass-dependent)
    
    vec3 color;
    float alpha;
    
    if (vRadius >= 5.0) {
        // ===== SUPERMASSIVE BLACK HOLE =====
        // Blazing white core with golden accretion disk halo
        float core = exp(-distSq * 8.0);
        float halo = exp(-distSq * 1.5);
        float outerGlow = exp(-distSq * 0.5);
        
        vec3 coreColor = vec3(1.0, 1.0, 1.0);
        vec3 haloColor = vec3(1.0, 0.8, 0.3);
        vec3 outerColor = vec3(0.8, 0.4, 0.1);
        
        color = coreColor * core + haloColor * halo * 0.8 + outerColor * outerGlow * 0.4;
        alpha = max(core, max(halo * 0.9, outerGlow * 0.5));
    }
    else if (vRadius > 1.2) {
        // ===== SPAWNED / MASSIVE BODY =====
        // Electric cyan/blue with energetic glow
        float core = exp(-distSq * 6.0);
        float glow = exp(-distSq * 2.0);
        float outerGlow = exp(-distSq * 0.8);
        
        float t = clamp((vRadius - 1.2) / 4.0, 0.0, 1.0);
        vec3 smallColor = vec3(0.1, 0.6, 1.0);  // electric blue
        vec3 bigColor   = vec3(0.9, 0.2, 0.5);  // hot pink/magenta for very massive
        vec3 baseColor = mix(smallColor, bigColor, t);
        
        color = vec3(1.0) * core * 0.8 + baseColor * glow + baseColor * 0.3 * outerGlow;
        alpha = max(core, max(glow * 0.8, outerGlow * 0.3));
    }
    else {
        // ===== GALAXY STARS (bulge + spiral arms) =====
        // Milky Way color gradient:
        //   Center bulge → warm golden/amber (old population II stars)
        //   Mid disk → warm white (Sun-like stars)
        //   Outer arms → blue-white (young hot OB stars + star-forming regions)
        float posHash = fract(sin(dot(vWorldPos * 0.01, vec2(12.9898, 78.233))) * 43758.5453);
        
        // Distance from galactic center (normalized to galaxy radius)
        float galacticDist = length(vWorldPos);
        float t = clamp(galacticDist / 600.0 + posHash * 0.15 - 0.075, 0.0, 1.0);
        
        // Color palette inspired by real Milky Way observations
        vec3 bulgeColor = vec3(1.0, 0.7, 0.35);    // warm golden (bulge)
        vec3 diskColor  = vec3(1.0, 0.92, 0.78);    // warm white (disk)
        vec3 armColor   = vec3(0.65, 0.82, 1.0);    // cool blue-white (young stars)
        
        vec3 starColor;
        if (t < 0.35) {
            starColor = mix(bulgeColor, diskColor, t / 0.35);
        } else {
            starColor = mix(diskColor, armColor, (t - 0.35) / 0.65);
        }
        
        // Soft circular glow (galaxy stars are tiny, so the glow IS the star)
        float core = exp(-distSq * 4.0);
        float glow = exp(-distSq * 1.5);
        
        color = starColor * core * 1.2 + starColor * 0.6 * glow;
        alpha = core * 0.9 + glow * 0.35;
        
        // Subtle twinkle effect 
        float twinkle = 0.9 + 0.1 * sin(uTime * 2.5 + posHash * 6.28);
        color *= twinkle;
        alpha *= twinkle;
    }
    
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
    glfwWindowHint(GLFW_SAMPLES, 4); // MSAA for smoother points

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
    // Additive blending: overlapping stars create a brighter glow (realistic galaxy look)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glEnable(GL_MULTISAMPLE);

    // Register mouse callback and store 'this' pointer for access from the static callback
    glfwSetWindowUserPointer(window, this);
    glfwSetMouseButtonCallback(window, Renderer::mouseButtonCallback);
    // Register scroll callback for zoom
    glfwSetScrollCallback(window, Renderer::scrollCallback);

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

    // Deep space background with very subtle blue tint
    glClearColor(0.005f, 0.005f, 0.015f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(shaderProgram);

    // Process basic camera input
    processInput();

    // Set Uniforms
    glUniform2f(glGetUniformLocation(shaderProgram, "cameraPos"), cameraX, cameraY);
    glUniform1f(glGetUniformLocation(shaderProgram, "cameraZoom"), cameraZoom);
    glUniform2f(glGetUniformLocation(shaderProgram, "resolution"), (float)width, (float)height);
    
    // Time uniform for animation effects
    float time = (float)glfwGetTime();
    glUniform1f(glGetUniformLocation(shaderProgram, "uTime"), time);

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
    
    // R key to reset camera
    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
        cameraX = 0; cameraY = 0; cameraZoom = 0.001f;
    }
}

GLuint Renderer::compileShaders() {
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    
    // Check vertex shader compile errors
    GLint success;
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[512];
        glGetShaderInfoLog(vertexShader, 512, NULL, log);
        std::cerr << "Vertex shader compile error: " << log << std::endl;
    }

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    
    // Check fragment shader compile errors
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[512];
        glGetShaderInfoLog(fragmentShader, 512, NULL, log);
        std::cerr << "Fragment shader compile error: " << log << std::endl;
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);
    
    // Check link errors
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char log[512];
        glGetProgramInfoLog(program, 512, NULL, log);
        std::cerr << "Shader link error: " << log << std::endl;
    }

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

void Renderer::scrollCallback(GLFWwindow* window, double xoff, double yoff) {
    Renderer* self = static_cast<Renderer*>(glfwGetWindowUserPointer(window));
    if (!self) return;
    
    float zoomFactor = 1.1f;
    if (yoff > 0) {
        self->cameraZoom *= zoomFactor;
    } else if (yoff < 0) {
        self->cameraZoom /= zoomFactor;
    }
}
