cmake_minimum_required(VERSION 3.18)
project(DynaMeshThermal LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Suppress some CUDA warnings for cleaner output and force standard preprocessor for CCCL
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler /W3,/Zc:preprocessor")

include_directories(${CMAKE_SOURCE_DIR}/src)

add_executable(DynaMeshThermal 
    src/main.cu
    src/math/LinearSolver.cu
    src/physics/NavierStokesDarcy.cu
    src/physics/AdvectionDiffusion.cu
    src/optimization/SIMP.cu
    src/optimization/Objective.cu
    src/optimization/Adjoint.cu
)

# Link cuSPARSE and cuBLAS for Krylov solvers
target_link_libraries(DynaMeshThermal cusparse cublas)

# AmgX link target placeholder
# target_link_libraries(DynaMeshThermal AmgX::amgx)
