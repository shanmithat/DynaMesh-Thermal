#pragma once
#include "MemoryManager.cuh"
#include <array>
#include <vector>
#include <memory>

namespace dynamesh {

// Dim = 2 or 3
// Staggered Grid (MAC Grid) abstraction
template <int Dim>
class MACGrid {
public:
    // nx, ny, nz
    std::array<int, Dim> resolution;
    std::array<float, Dim> spacing;

    // Number of cells
    size_t numCells;
    
    // Total sizes for staggered velocities
    std::array<size_t, Dim> numFaces;

    // Data buffers (Unified memory for fully coalesced access and host initialization)
    // Scalars (pressure, temperature, density - at cell centers)
    DeviceBuffer<float> pressure;
    DeviceBuffer<float> temperature;
    DeviceBuffer<float> density; // SIMP design variable

    // Velocities (Staggered at cell faces)
    // For 2D: u has nx+1, ny faces; v has nx, ny+1 faces
    // For 3D: u has nx+1, ny, nz faces; etc.
    std::vector<std::unique_ptr<DeviceBuffer<float>>> velocity;

    MACGrid(const std::array<int, Dim>& res, const std::array<float, Dim>& dx) 
        : resolution(res), spacing(dx), 
          numCells(1), 
          pressure(0), temperature(0), density(0)
    {
        for (int i = 0; i < Dim; ++i) {
            numCells *= resolution[i];
        }

        pressure = DeviceBuffer<float>(numCells);
        temperature = DeviceBuffer<float>(numCells);
        density = DeviceBuffer<float>(numCells);

        for (int i = 0; i < Dim; ++i) {
            size_t faceCount = 1;
            for (int j = 0; j < Dim; ++j) {
                if (i == j) faceCount *= (resolution[j] + 1);
                else faceCount *= resolution[j];
            }
            numFaces[i] = faceCount;
            velocity.push_back(std::make_unique<DeviceBuffer<float>>(faceCount));
        }
    }

    ~MACGrid() = default;

    // Helper functions for indexing
    // 2D Cell Index
    __host__ __device__ static inline size_t cellIdx2D(int i, int j, int nx) {
        return j * nx + i;
    }

    // 3D Cell Index
    __host__ __device__ static inline size_t cellIdx3D(int i, int j, int k, int nx, int ny) {
        return k * nx * ny + j * nx + i;
    }
    
    // Staggered face index functions would go here...
};

} // namespace dynamesh
