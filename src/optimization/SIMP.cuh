#pragma once
#include "../core/Grid.cuh"
#include "../core/Config.h"
#include <memory>

namespace dynamesh {
namespace optimization {

class SIMP {
public:
    SIMP(const Config& config, std::shared_ptr<MACGrid<2>> grid);
    ~SIMP();

    // Filters the design variables (density) using a spatial radius R
    void applyDensityFilter(float radiusR);

    // Placeholder for optimization update step (MMA / OC)
    void updateDesignVariables(float moveLimit);

private:
    Config config_;
    std::shared_ptr<MACGrid<2>> grid_;

    // Filter buffer to prevent in-place overwriting
    DeviceBuffer<float> filteredDensity_;
};

} // namespace optimization
} // namespace dynamesh
