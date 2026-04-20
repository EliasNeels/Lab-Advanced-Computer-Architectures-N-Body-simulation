#include "../include/renderer.h"
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cmath>

extern "C" void launchPackVBO(const Bodies& bodies, float* d_vbo, cudaStream_t stream);

// =====================================================================
// 3D Vertex Shader: Perspective projection with orbital camera
// =====================================================================
const char* vertexShaderSource = R"(
#version 450 core
layout (location = 0) in vec4 aData; // x, y, z, radius

uniform mat4 uMVP;
uniform float uScreenHeight;

out float vRadius;
out vec3 vWorldPos;

void main() {
    vec4 worldPos = vec4(aData.xyz, 1.0);
    gl_Position = uMVP * worldPos;
    
    // Scale point size by inverse distance (perspective effect)
    float dist = gl_Position.w;
    float pointSize = aData.w * uScreenHeight * 0.3 / max(dist, 1.0);
    gl_PointSize = clamp(pointSize, 1.5, 128.0);
    
    vRadius = aData.w;
    vWorldPos = aData.xyz;
}
)";

// =====================================================================
// 3D Fragment Shader: Glowing stars with spectral coloring
// =====================================================================
const char* fragmentShaderSource = R"(
#version 450 core
out vec4 FragColor;

in float vRadius;
in vec3 vWorldPos;

uniform float uTime;

void main() {
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    float distSq = dot(coord, coord);
    
    if (distSq > 1.0) discard;
    
    float dist = sqrt(distSq);
    
    vec3 color;
    float alpha;
    
    if (vRadius >= 5.0) {
        // ===== SUPERMASSIVE BLACK HOLE =====
        float core = exp(-distSq * 8.0);
        float halo = exp(-distSq * 1.5);
        float outerGlow = exp(-distSq * 0.5);
        
        color = vec3(1.0) * core + vec3(1.0, 0.8, 0.3) * halo * 0.8 + vec3(0.8, 0.4, 0.1) * outerGlow * 0.4;
        alpha = max(core, max(halo * 0.9, outerGlow * 0.5));
    }
    else if (vRadius > 1.2) {
        // ===== SPAWNED / MASSIVE BODY =====
        float core = exp(-distSq * 6.0);
        float glow = exp(-distSq * 2.0);
        float outerGlow = exp(-distSq * 0.8);
        
        float t = clamp((vRadius - 1.2) / 4.0, 0.0, 1.0);
        vec3 smallColor = vec3(0.1, 0.6, 1.0);
        vec3 bigColor   = vec3(0.9, 0.2, 0.5);
        vec3 baseColor = mix(smallColor, bigColor, t);
        
        color = vec3(1.0) * core * 0.8 + baseColor * glow + baseColor * 0.3 * outerGlow;
        alpha = max(core, max(glow * 0.8, outerGlow * 0.3));
    }
    else {
        // ===== GALAXY STARS =====
        float posHash = fract(sin(dot(vWorldPos.xy * 0.01, vec2(12.9898, 78.233))) * 43758.5453);
        
        float galacticDist = length(vWorldPos);
        float t = clamp(galacticDist / 600.0 + posHash * 0.15 - 0.075, 0.0, 1.0);
        
        vec3 bulgeColor = vec3(1.0, 0.7, 0.35);
        vec3 diskColor  = vec3(1.0, 0.92, 0.78);
        vec3 armColor   = vec3(0.65, 0.82, 1.0);
        
        vec3 starColor;
        if (t < 0.35) {
            starColor = mix(bulgeColor, diskColor, t / 0.35);
        } else {
            starColor = mix(diskColor, armColor, (t - 0.35) / 0.65);
        }
        
        float core = exp(-distSq * 4.0);
        float glow = exp(-distSq * 1.5);
        
        color = starColor * core * 1.2 + starColor * 0.6 * glow;
        alpha = core * 0.9 + glow * 0.35;
        
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
      maxBodies(1000000),
      camTargetX(0), camTargetY(0), camTargetZ(0),
      camDistance(1500.0f), camTheta(0.0f), camPhi(0.4f),
      mouseRotating(false), lastMouseX(0), lastMouseY(0)
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
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return false;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);

    window = glfwCreateWindow(windowWidth, windowHeight, "CUDA N-Body 3D", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create window" << std::endl;
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return false;
    }

    shaderProgram = compileShaders();

    cudaMalloc(&d_vbo_data, maxBodies * sizeof(float) * 4);
    h_vbo_data = new float[maxBodies * 4];
    
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, maxBodies * sizeof(float) * 4, nullptr, GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE); // Additive blending
    glEnable(GL_MULTISAMPLE);
    // No depth test — additive blending particles don't need it
    glDisable(GL_DEPTH_TEST);

    glfwSetWindowUserPointer(window, this);
    glfwSetMouseButtonCallback(window, Renderer::mouseButtonCallback);
    glfwSetScrollCallback(window, Renderer::scrollCallback);

    return true;
}

void Renderer::updateVBO(Bodies& bodies, cudaStream_t stream) {
    if (bodies.count == 0) return;

    launchPackVBO(bodies, d_vbo_data, stream);
    cudaMemcpyAsync(h_vbo_data, d_vbo_data, bodies.count * sizeof(float) * 4, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, bodies.count * sizeof(float) * 4, h_vbo_data);
}

// =====================================================================
// Simple 4x4 matrix math (avoiding glm dependency)
// =====================================================================
struct Mat4 {
    float m[16];
};

static Mat4 mat4Identity() {
    Mat4 r = {};
    r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;
    return r;
}

static Mat4 mat4Multiply(const Mat4& a, const Mat4& b) {
    Mat4 r = {};
    // Column-major multiply: R[row][col] = sum_k A[row][k] * B[k][col]
    for (int col = 0; col < 4; col++)
        for (int row = 0; row < 4; row++)
            for (int k = 0; k < 4; k++)
                r.m[col * 4 + row] += a.m[k * 4 + row] * b.m[col * 4 + k];
    return r;
}

static Mat4 mat4Perspective(float fovY, float aspect, float nearZ, float farZ) {
    Mat4 r = {};
    float f = 1.0f / tanf(fovY * 0.5f);
    r.m[0] = f / aspect;
    r.m[5] = f;
    r.m[10] = (farZ + nearZ) / (nearZ - farZ);
    r.m[11] = -1.0f;
    r.m[14] = (2.0f * farZ * nearZ) / (nearZ - farZ);
    return r;
}

static Mat4 mat4LookAt(float eyeX, float eyeY, float eyeZ,
                       float centerX, float centerY, float centerZ,
                       float upX, float upY, float upZ) {
    float fx = centerX - eyeX, fy = centerY - eyeY, fz = centerZ - eyeZ;
    float flen = sqrtf(fx*fx + fy*fy + fz*fz);
    fx /= flen; fy /= flen; fz /= flen;
    
    // s = f × up
    float sx = fy * upZ - fz * upY;
    float sy = fz * upX - fx * upZ;
    float sz = fx * upY - fy * upX;
    float slen = sqrtf(sx*sx + sy*sy + sz*sz);
    sx /= slen; sy /= slen; sz /= slen;
    
    // u = s × f
    float ux = sy * fz - sz * fy;
    float uy = sz * fx - sx * fz;
    float uz = sx * fy - sy * fx;
    
    Mat4 r = mat4Identity();
    r.m[0] = sx;  r.m[4] = sy;  r.m[8]  = sz;
    r.m[1] = ux;  r.m[5] = uy;  r.m[9]  = uz;
    r.m[2] = -fx; r.m[6] = -fy; r.m[10] = -fz;
    r.m[12] = -(sx * eyeX + sy * eyeY + sz * eyeZ);
    r.m[13] = -(ux * eyeX + uy * eyeY + uz * eyeZ);
    r.m[14] = (fx * eyeX + fy * eyeY + fz * eyeZ);
    return r;
}

void Renderer::render(int bodyCount) {
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    glViewport(0, 0, width, height);

    glClearColor(0.005f, 0.005f, 0.015f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(shaderProgram);

    processInput();

    // Build MVP from orbital camera
    float aspect = (float)width / (float)height;
    Mat4 proj = mat4Perspective(0.9f, aspect, 1.0f, 100000.0f);  // ~51° FOV
    
    // Spherical to Cartesian for eye position
    float eyeX = camTargetX + camDistance * cosf(camPhi) * sinf(camTheta);
    float eyeY = camTargetY + camDistance * sinf(camPhi);
    float eyeZ = camTargetZ + camDistance * cosf(camPhi) * cosf(camTheta);
    
    Mat4 view = mat4LookAt(eyeX, eyeY, eyeZ, camTargetX, camTargetY, camTargetZ, 0, 1, 0);
    Mat4 mvp = mat4Multiply(proj, view);

    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "uMVP"), 1, GL_FALSE, mvp.m);
    glUniform1f(glGetUniformLocation(shaderProgram, "uScreenHeight"), (float)height);
    
    float time = (float)glfwGetTime();
    glUniform1f(glGetUniformLocation(shaderProgram, "uTime"), time);

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
    
    float speed = camDistance * 0.01f;
    
    // WASD pans the look-at target
    // Compute camera-relative forward/right on the XZ plane
    float fwdX = sinf(camTheta), fwdZ = cosf(camTheta);
    float rightX = cosf(camTheta), rightZ = -sinf(camTheta);
    
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) { camTargetX += fwdX * speed * 0.016f; camTargetZ += fwdZ * speed * 0.016f; }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) { camTargetX -= fwdX * speed * 0.016f; camTargetZ -= fwdZ * speed * 0.016f; }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) { camTargetX -= rightX * speed * 0.016f; camTargetZ -= rightZ * speed * 0.016f; }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) { camTargetX += rightX * speed * 0.016f; camTargetZ += rightZ * speed * 0.016f; }
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) camTargetY += speed * 0.016f;
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) camTargetY -= speed * 0.016f;
    
    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) camDistance *= 0.98f;
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) camDistance *= 1.02f;
    
    // R = reset camera
    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
        camTargetX = 0; camTargetY = 0; camTargetZ = 0;
        camDistance = 1500.0f; camTheta = 0.0f; camPhi = 0.4f;
    }
    
    // Right-click drag to rotate camera
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
        double mx, my;
        glfwGetCursorPos(window, &mx, &my);
        if (!mouseRotating) {
            mouseRotating = true;
            lastMouseX = mx;
            lastMouseY = my;
        } else {
            float dx = (float)(mx - lastMouseX) * 0.005f;
            float dy = (float)(my - lastMouseY) * 0.005f;
            camTheta -= dx;
            camPhi += dy;
            // Clamp phi to avoid gimbal lock (keep between -89° and +89°)
            camPhi = fmaxf(-1.5f, fminf(1.5f, camPhi));
            lastMouseX = mx;
            lastMouseY = my;
        }
    } else {
        mouseRotating = false;
    }
}

GLuint Renderer::compileShaders() {
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    
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
    // For 3D, project onto the XY plane at z=0
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    
    // Simple approximation: map screen coords relative to camera target
    float ndcX = (2.0f * (float)screenX / width - 1.0f);
    float ndcY = -(2.0f * (float)screenY / height - 1.0f);
    
    float fov = 0.9f;
    float aspect = (float)width / (float)height;
    worldX = camTargetX + ndcX * camDistance * tanf(fov * 0.5f) * aspect;
    worldY = camTargetY + ndcY * camDistance * tanf(fov * 0.5f);
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
        self->camDistance /= zoomFactor;
    } else if (yoff < 0) {
        self->camDistance *= zoomFactor;
    }
    // Clamp distance
    self->camDistance = fmaxf(10.0f, fminf(50000.0f, self->camDistance));
}
