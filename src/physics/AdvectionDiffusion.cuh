#pragma once
#include "../core/Grid.cuh"
#include "../core/Config.h"
#include <memory>

namespace dynamesh {
namespace physics {

class AdvectionDiffusion {
public:
    AdvectionDiffusion(const Config& config, std::shared_ptr<MACGrid<2>> grid);
    ~AdvectionDiffusion();

    // Assembles the sparse linear system for the temperature field
    void assembleSystem();
    
    // Solves the steady state temperature field
    void solve();

private:
    Config config_;
    std::shared_ptr<MACGrid<2>> grid_;

    // CSR Matrix data for temperature solve
    int* csrRowPtr_;
    int* csrColInd_;
    float* csrVal_;
};

} // namespace physics
} // namespace dynamesh
