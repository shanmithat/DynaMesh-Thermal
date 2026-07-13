#pragma once
#include <vector>
#include <memory>

// Forward declarations for opaque handles
struct cusparseContext;
struct cublasContext;

namespace dynamesh {

// Abstract interface for linear solvers to allow swapping between custom cuSPARSE and AmgX
class ILinearSolver {
public:
    virtual ~ILinearSolver() = default;

    // Initialize the solver with matrix structure
    virtual void setup(int numRows, int numNonZeros, const int* csrRowPtr, const int* csrColInd) = 0;
    
    // Update matrix values (for non-linear or time-dependent problems)
    virtual void updateValues(const float* csrVal) = 0;
    
    // Solve Ax = b
    virtual void solve(const float* b, float* x) = 0;
};

class CuSparseKrylovSolver : public ILinearSolver {
public:
    enum class Method {
        BiCGStab,
        GMRES
    };

    CuSparseKrylovSolver(Method method = Method::BiCGStab);
    ~CuSparseKrylovSolver() override;

    void setup(int numRows, int numNonZeros, const int* csrRowPtr, const int* csrColInd) override;
    void updateValues(const float* csrVal) override;
    void solve(const float* b, float* x) override;

private:
    Method method_;
    int numRows_;
    int numNonZeros_;
    
    // Matrix structure
    const int* csrRowPtr_;
    const int* csrColInd_;
    const float* csrVal_;
    
    // Preconditioner data
    float* iluVal_;
    
    // cuSPARSE and cuBLAS handles
    cusparseContext* cusparseHandle_;
    cublasContext* cublasHandle_;
    
    // Opaque descriptors for cuSPARSE
    void* matA_;
    void* matM_; // Preconditioner
    void* matILU_; // ILU info
    
    // Helper methods for ILU computation and Krylov loop
    void computeILU();
    void solveBiCGStab(const float* b, float* x);
    void solveGMRES(const float* b, float* x);
};

} // namespace dynamesh
