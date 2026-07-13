#include "SIMP.cuh"
#include <cmath>

namespace dynamesh {
namespace optimization {

__global__ void densityFilterKernel(
    int nx, int ny, float dx, float dy, float radiusR,
    const float* densityIn, float* densityOut) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < nx && j < ny) {
        float sumWeights = 0.0f;
        float sumFiltered = 0.0f;
        
        // Define bounding box for the filter
        int radiusCellsX = ceilf(radiusR / dx);
        int radiusCellsY = ceilf(radiusR / dy);

        int minX = max(0, i - radiusCellsX);
        int maxX = min(nx - 1, i + radiusCellsX);
        int minY = max(0, j - radiusCellsY);
        int maxY = min(ny - 1, j + radiusCellsY);

        for (int q = minY; q <= maxY; ++q) {
            for (int p = minX; p <= maxX; ++p) {
                float dist = sqrtf(powf((i - p) * dx, 2.0f) + powf((j - q) * dy, 2.0f));
                if (dist <= radiusR) {
                    float weight = radiusR - dist; // Linear cone filter
                    sumWeights += weight;
                    sumFiltered += weight * densityIn[q * nx + p];
                }
            }
        }
        
        densityOut[j * nx + i] = sumFiltered / sumWeights;
    }
}

SIMP::SIMP(const Config& config, std::shared_ptr<MACGrid<2>> grid)
    : config_(config), grid_(grid), filteredDensity_(grid->numCells) 
{
    // Initialize densities to uniform volume fraction
    float initVal = config_.volumeFraction;
    // Launch a simple kernel to fill density (skipped here for brevity, assume filled)
}

SIMP::~SIMP() {}

void SIMP::applyDensityFilter(float radiusR) {
    dim3 blockSize(16, 16);
    dim3 gridSize((config_.nx + blockSize.x - 1) / blockSize.x, 
                  (config_.ny + blockSize.y - 1) / blockSize.y);
                  
    densityFilterKernel<<<gridSize, blockSize>>>(
        config_.nx, config_.ny, config_.dx, config_.dy, radiusR,
        grid_->density.data(), filteredDensity_.data()
    );
    
    // Copy filtered back to main density buffer
    gpuErrchk(cudaMemcpy(grid_->density.data(), filteredDensity_.data(), 
                         grid_->numCells * sizeof(float), cudaMemcpyDeviceToDevice));
}

void SIMP::updateDesignVariables(float moveLimit) {
    // Implement OC or MMA logic here using sensitivities
}

} // namespace optimization
} // namespace dynamesh
