#pragma once

namespace dynamesh {

struct Config {
    // Grid settings
    int dim = 2;
    int nx = 128;
    int ny = 64;
    int nz = 1;
    float dx = 0.01f;
    float dy = 0.01f;
    float dz = 0.01f;

    // Physical parameters
    float reynoldsNumber = 100.0f;
    float prandtlNumber = 1.0f;
    float thermalConductivity = 1.0f;

    // SIMP & Darcy penalty parameters
    float simpPenaltyPower = 3.0f;
    float darcyMinPenalty = 0.0f;
    float darcyMaxPenalty = 1e5f;
    
    // Optimization parameters
    float volumeFraction = 0.5f;
    int maxIterations = 200;
};

} // namespace dynamesh
