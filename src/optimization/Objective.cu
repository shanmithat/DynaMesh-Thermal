#include "Objective.cuh"
#include <cub/cub.cuh>

namespace dynamesh {
namespace optimization {

Objective::Objective(const Config& config, std::shared_ptr<MACGrid<2>> grid)
    : config_(config), grid_(grid) 
{
    // CUB will require temporary storage, we allocate a small buffer
    // For simplicity in this structure, we just leave it for the method to allocate
}

Objective::~Objective() {}

float Objective::computeThermalCompliance() {
    float* d_out = nullptr;
    gpuErrchk(cudaMalloc(&d_out, sizeof(float)));

    // Determine temporary device storage requirements for CUB
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    
    // We want to sum the temperature array as a basic thermal compliance proxy
    // (Assuming uniform heat source field = 1.0)
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, 
                           grid_->temperature.data(), d_out, grid_->numCells);
                           
    // Allocate temporary storage
    gpuErrchk(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    
    // Run reduction
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, 
                           grid_->temperature.data(), d_out, grid_->numCells);

    // Fetch result
    float h_out = 0.0f;
    gpuErrchk(cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    
    gpuErrchk(cudaFree(d_temp_storage));
    gpuErrchk(cudaFree(d_out));
    
    // Scale by cell volume
    return h_out * config_.dx * config_.dy;
}

} // namespace optimization
} // namespace dynamesh
