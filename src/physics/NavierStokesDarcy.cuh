#pragma once
#include "../core/Grid.cuh"
#include "../core/Config.h"
#include <memory>

namespace dynamesh {
namespace physics {

class NavierStokesDarcy {
public:
    NavierStokesDarcy(const Config& config, std::shared_ptr<MACGrid<2>> grid);
    ~NavierStokesDarcy();

    // Assembles the sparse linear system mapping velocities and pressures
    // Incorporates the SIMP density (gamma) via the Darcy porosity penalty
    void assembleSystem();
    
    // Solves the steady state velocity field
    void solve();

private:
    Config config_;
    std::shared_ptr<MACGrid<2>> grid_;

    // CSR Matrix data for velocity-pressure coupling
    // In practice, we solve for u, v, p simultaneously or via projection method.
    // For steady state, assembling the full block system or segregated SIMPLE approach is used.
    int* csrRowPtr_;
    int* csrColInd_;
    float* csrVal_;
};

} // namespace physics
} // namespace dynamesh
