#pragma once
#include "../core/Grid.cuh"
#include "../core/Config.h"
#include "../math/LinearSolver.cuh"
#include <memory>

namespace dynamesh {
namespace optimization {

class Adjoint {
public:
    Adjoint(const Config& config, std::shared_ptr<MACGrid<2>> grid);
    ~Adjoint();

    // Assemble the transposed sparse matrix system for Advection-Diffusion
    // and compute the RHS: -df/dT
    void assembleSystem();
    
    // Solves the Adjoint field lambda
    void solve();
    
    // Computes the objective gradient df/dgamma
    void computeSensitivities(float* sensitivitiesOut);

private:
    Config config_;
    std::shared_ptr<MACGrid<2>> grid_;

    // Linear solver for A^T * lambda = RHS
    std::unique_ptr<ILinearSolver> adjointSolver_;

    // Adjoint variable (lambda) buffer
    DeviceBuffer<float> lambda_;
    
    // RHS vector
    DeviceBuffer<float> rhs_;
};

} // namespace optimization
} // namespace dynamesh
