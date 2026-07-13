#include <iostream>
#include "core/MemoryManager.cuh"
#include "core/Grid.cuh"
#include "physics/NavierStokesDarcy.cuh"
#include "physics/AdvectionDiffusion.cuh"
#include "optimization/SIMP.cuh"
#include "optimization/Objective.cuh"
#include "optimization/Adjoint.cuh"


int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "[FATAL] No CUDA-capable devices found." << std::endl;
        return -1;
    }
    cudaSetDevice(0); // Explicitly bind thread context to device 0
    std::cout << "DynaMesh-Thermal: 2D/3D Topology Optimization Engine" << std::endl;
    
    // Configuration
    dynamesh::Config config;
    config.nx = 128;
    config.ny = 64;
    config.maxIterations = 50;
    
    // Allocate MAC Grid
    std::array<int, 2> res2D = {config.nx, config.ny};
    std::array<float, 2> dx2D = {config.dx, config.dy};
    auto grid2D = std::make_shared<dynamesh::MACGrid<2>>(res2D, dx2D);
    
    // Initialize Physics & Optimization Modules
    dynamesh::physics::NavierStokesDarcy nsSolver(config, grid2D);
    dynamesh::physics::AdvectionDiffusion adSolver(config, grid2D);
    dynamesh::optimization::SIMP simpOpt(config, grid2D);
    dynamesh::optimization::Objective objCalc(config, grid2D);
    dynamesh::optimization::Adjoint adjointSolver(config, grid2D);

    std::cout << "Starting Optimization Loop (" << config.maxIterations << " iterations max)..." << std::endl;
    
    for (int iter = 0; iter < config.maxIterations; ++iter) {
        // 1. Filter Design Variables
        simpOpt.applyDensityFilter(1.5f * config.dx);
        
        // 2. Solve Forward Physics
        nsSolver.assembleSystem();
        nsSolver.solve();
        
        adSolver.assembleSystem();
        adSolver.solve();
        
        // 3. Evaluate Objective
        float compliance = objCalc.computeThermalCompliance();
        std::cout << "Iter: " << iter << " | Thermal Compliance: " << compliance << std::endl;
        
        // 4. Solve Adjoint & Compute Sensitivities
        adjointSolver.assembleSystem();
        adjointSolver.solve();
        
        // 5. Update Design Variables
        simpOpt.updateDesignVariables(0.2f); // Move limit
    }
    
    std::cout << "Optimization loop completed successfully." << std::endl;
    return 0;
}
