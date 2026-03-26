#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <cuda_runtime.h>

int main() {
    std::cout << "Hello World from N-Body Simulation!" << std::endl;

    // Test C++ std
    std::cout << "Testing C++ standard 20 features (e.g., concepts, numbers)..." << std::endl;

    // Test CUDA
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess) {
        std::cout << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
    } else {
        std::cout << "CUDA Devices found: " << deviceCount << std::endl;
    }

    // Test GLFW
    if (!glfwInit()) {
        std::cout << "Failed to initialize GLFW" << std::endl;
    } else {
        std::cout << "GLFW initialized successfully!" << std::endl;
        glfwTerminate();
    }

    return 0;
}