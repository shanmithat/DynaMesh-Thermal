#pragma once
#include "../core/Grid.cuh"
#include "../core/Config.h"
#include <memory>

namespace dynamesh {
namespace optimization {

class Objective {
public:
    Objective(const Config& config, std::shared_ptr<MACGrid<2>> grid);
    ~Objective();

    // Compute thermal compliance (e.g. integral of Temperature * Heat Source)
    // Uses parallel reduction on the GPU
    float computeThermalCompliance();

private:
    Config config_;
    std::shared_ptr<MACGrid<2>> grid_;

    // Buffer for intermediate reduction results
    DeviceBuffer<float> d_reductionBuffer_;
};

} // namespace optimization
} // namespace dynamesh
