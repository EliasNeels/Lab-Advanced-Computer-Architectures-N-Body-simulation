cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(NBodySimulation LANGUAGES CXX CUDA)

# --- Configuration ---
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Optimization flags
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  add_compile_options(-O3)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-O3;--use_fast_math")
else()
  add_compile_options(-g)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-g;-G;-lineinfo")
endif()

# Autodetect CUDA architecture if not set
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86) # Matches RTX 3050 Ti
endif()

# --- Find Dependencies ---
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)
find_package(CUDAToolkit REQUIRED)

# --- Manage External Dependencies via FetchContent ---
include(FetchContent)

# Dear ImGui
FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG        v1.90.4
)
FetchContent_MakeAvailable(imgui)

# --- Prepare Sources ---
# For initial testing, we only use main.cpp to avoid errors from incomplete files
set(SOURCES 
    "src/main.cpp"
    "src/renderer.cpp"
    "cuda/bounding_box.cu"
    "cuda/morton_sort.cu"
    "cuda/build_tree_and_com.cu"
    "cuda/force_calculation.cu"
    "cuda/integration.cu"
    "cuda/collision.cu"
    "cuda/graphics_interop.cu"
)

file(GLOB IMGUI_SOURCES 
    "${imgui_SOURCE_DIR}/imgui.cpp"
    "${imgui_SOURCE_DIR}/imgui_draw.cpp"
    "${imgui_SOURCE_DIR}/imgui_tables.cpp"
    "${imgui_SOURCE_DIR}/imgui_widgets.cpp"
    "${imgui_SOURCE_DIR}/imgui_demo.cpp"
    "${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp"
    "${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp"
)

# --- Target ---
add_executable(${PROJECT_NAME} ${SOURCES} ${IMGUI_SOURCES})

# --- Include Directories ---
target_include_directories(${PROJECT_NAME} PRIVATE 
    include
    cuda
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
    ${CUDAToolkit_INCLUDE_DIRS}
)

# --- Link Libraries ---
target_link_libraries(${PROJECT_NAME} PRIVATE 
    GLEW::GLEW
    glfw
    OpenGL::GL
    CUDA::cudart
)

# --- Extra Target Properties ---
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)