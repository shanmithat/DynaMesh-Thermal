#include "Adjoint.cuh"

namespace dynamesh {
namespace optimization {

__global__ void computeRHSThermalCompliance(
    int numCells, float cellVolume, float* rhs) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numCells) {
        // For compliance f = T^T * Q, df/dT = Q. 
        // Assuming uniform Q = 1.0
        rhs[i] = -1.0f * cellVolume; 
    }
}

__global__ void computeSensitivitiesKernel(
    int numCells, const float* lambda, const float* temperature, float* sensitivities) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numCells) {
        // Evaluate df/dgamma = lambda^T * (dA/dgamma) * T
        // Placeholder for sensitivity computation
        sensitivities[i] = 0.0f; // Compute exact derivative via chain rule
    }
}

Adjoint::Adjoint(const Config& config, std::shared_ptr<MACGrid<2>> grid)
    : config_(config), grid_(grid), 
      lambda_(grid->numCells), rhs_(grid->numCells) 
{
    adjointSolver_ = std::make_unique<CuSparseKrylovSolver>(CuSparseKrylovSolver::Method::BiCGStab);
    // Setup solver structure here...
}

Adjoint::~Adjoint() {}

void Adjoint::assembleSystem() {
    // 1. Build A^T (Transposed Advection-Diffusion matrix)
    // 2. Compute RHS (-df/dT)
    size_t numCells = grid_->numCells;
    int threads = 256;
    int blocks = (int)((numCells + threads - 1) / threads);
    
    float cellVolume = config_.dx * config_.dy;
    computeRHSThermalCompliance<<<blocks, threads>>>((int)numCells, cellVolume, rhs_.data());
}

void Adjoint::solve() {
    // Call solver on A^T * lambda = RHS
    // adjointSolver_->solve(rhs_.data(), lambda_.data());
}

void Adjoint::computeSensitivities(float* sensitivitiesOut) {
    size_t numCells = grid_->numCells;
    int threads = 256;
    int blocks = (int)((numCells + threads - 1) / threads);
    
    computeSensitivitiesKernel<<<blocks, threads>>>(
        numCells, lambda_.data(), grid_->temperature.data(), sensitivitiesOut
    );
}

} // namespace optimization
} // namespace dynamesh
