#include <RcppEigen.h>
#include <complex>
#include <random>
#include <fftw3.h>
#include "circulant_utils.h"

// [[Rcpp::depends(RcppEigen)]]

using namespace Rcpp;
using namespace Eigen;

// Function for log-density calculation
// [[Rcpp::export]]
Eigen::VectorXd dmatern_copula_circulant(const Eigen::MatrixXd& X, int dim1, int dim2, double rho1, double rho2, int nu) {
    int n_obs = X.cols();
    Eigen::MatrixXd c = create_base_matrix(dim2, dim1, rho2, rho1);
    Eigen::MatrixXcd eigs = compute_and_rescale_eigenvalues(c, nu);
    Eigen::VectorXd quad_forms(n_obs);
    
    // Compute log determinant
    double log_det = eigs.array().log().sum().real();
    
    // Compute quadratic form x^T Q x
    for (int i = 0; i < n_obs; ++i) {
        Eigen::VectorXd Qx = matrix_vector_product(eigs, X.col(i));
        quad_forms(i) = X.col(i).dot(Qx);
    }
    
    // Subtract x'Ix (sum of squared elements for each observation)
    Eigen::VectorXd x_squared = X.colwise().squaredNorm();

    // Compute log density
    Eigen::VectorXd log_densities = -0.5 * (quad_forms.array() - log_det - x_squared.array());
    
    return log_densities;
}

// [[Rcpp::export]]
Eigen::MatrixXd rmatern_copula_circulant(int n_samples, int dim1, int dim2, double rho1, double rho2, int nu) {
    Eigen::MatrixXd samples(dim1 * dim2, n_samples);

    // Step 1: Compute the (real) eigenvalues, Λ = √(nN) DFT2(θ)
    Eigen::MatrixXd c = create_base_matrix(dim1, dim2, rho1, rho2);
    Eigen::MatrixXcd eigs = compute_and_rescale_eigenvalues(c, nu);
    Eigen::MatrixXcd lambda_sqrt = eigs.array().sqrt().inverse();

    // Step 1: Sample z, where Re(z_ij) ~ N(0,1) and Im(z_ij) ~ N(0,1) iid
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 1);
    Eigen::MatrixXcd z(dim2, dim1);
    double scale = 1.0 / std::sqrt(dim1 * dim2);
    
    for(int s = 0; s < n_samples; ++s) {
        for(int i = 0; i < dim2; ++i) {
            for(int j = 0; j < dim1; ++j) {
                z(i, j) = std::complex<double>(d(gen), d(gen));
            }
        }
        
        Eigen::MatrixXcd v = fft2(lambda_sqrt.cwiseProduct(z), scale);
        
        // Step 4: x = Re(v)
        samples.col(s) = v.real().reshaped(v.size(), 1);
    }
    
    // Step 5: Return x
    return samples;
}

// [[Rcpp::export]]
Eigen::SparseMatrix<double> construct_circulant_precision(int dim1, int dim2, double rho1, double rho2, int nu) {
    Eigen::MatrixXd c = create_base_matrix(dim1, dim2, rho1, rho2);
    Eigen::MatrixXcd eigenvalues = compute_and_rescale_eigenvalues(c, nu);
    int N = dim1 * dim2;
    Eigen::SparseMatrix<double> Q(N, N);
    Q.reserve(Eigen::VectorXi::Constant(N, 9));  // Reserve space for 9 non-zeros per row

    fftw_complex *data = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);

    // Prepare input for IFFT
    for (int i = 0; i < dim2; ++i) {
        for (int j = 0; j < dim1; ++j) {
            std::complex<double> eig = eigenvalues(i, j);
            data[i*dim1 + j][0] = eig.real();
            data[i*dim1 + j][1] = eig.imag();
        }
    }

    // Perform in-place 2D IFFT
    fftw_plan plan = fftw_plan_dft_2d(dim2, dim1, data, data, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(plan);

    // Construct sparse precision matrix (only upper triangle due to symmetry)
    double threshold = 1e-10;
    for (int i = 0; i < N; ++i) {
        for (int j = i; j < N; ++j) {  // Only upper triangle
            int di = (i / dim1 - j / dim1 + dim2) % dim2;
            int dj = (i % dim1 - j % dim1 + dim1) % dim1;
            double value = data[di * dim1 + dj][0] / N;
            if (std::abs(value) > threshold) {
                Q.insert(i, j) = value;
                if (i != j) Q.insert(j, i) = value;  // Mirror for symmetry
            }
        }
    }

    fftw_destroy_plan(plan);
    fftw_free(data);

    Q.makeCompressed();
    return Q;
}