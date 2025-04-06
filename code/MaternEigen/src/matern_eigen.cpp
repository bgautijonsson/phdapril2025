#include <RcppEigen.h>
#include <omp.h>
#include <random>
#include "ar_matrix.h"


// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(openmp)]]

using namespace Rcpp;
using namespace Eigen;

// Function to compute the log-density of a Matérn-like field using eigendecomposition
// [[Rcpp::export]]
Eigen::VectorXd dmatern_eigen(const Eigen::MatrixXd& X, int dim_x, int dim_y, double rho1, double rho2, int nu) {
    int n_obs = X.cols();
    int N = dim_x * dim_y;
    ArrayXd quadform_sums = ArrayXd::Zero(n_obs);
    const double C = N * std::log(2 * M_PI);

    // Create precision matrices
    SparseMatrix<double> Q1 = make_AR_prec_matrix(dim_x, rho1);
    SparseMatrix<double> Q2 = make_AR_prec_matrix(dim_y, rho2);

    // Perform eigendecompositions
    SelfAdjointEigenSolver<SparseMatrix<double>> solver1(Q1);
    VectorXd A1 = solver1.eigenvalues();
    MatrixXd V1 = solver1.eigenvectors();

    SelfAdjointEigenSolver<SparseMatrix<double>> solver2(Q2);
    VectorXd A2 = solver2.eigenvalues();
    MatrixXd V2 = solver2.eigenvectors();


    double log_det = 0;

    #pragma omp parallel
    {
        ArrayXd local_quadform_sums = ArrayXd::Zero(n_obs);
        VectorXd v(N);
        VectorXd u(N);
        double lambda;

        #pragma omp for reduction(+:log_det)
        for (int i = 0; i < dim_x; ++i) {
            for (int j = 0; j < dim_y; ++j) {
                // First calculate the Kronecker product
                v = kroneckerProduct(V1.col(i), V2.col(j));
                
                // Compute the eigenvalue
                lambda = std::pow(A1(i) + A2(j), nu + 1);

                log_det += std::log(lambda);
                
                // Loop over observations
                for (int obs = 0; obs < n_obs; ++obs) {
                    double u = v.dot(X.col(obs));
                    local_quadform_sums(obs) += u * u * lambda;
                }
            }
        }

        #pragma omp critical
        {
            quadform_sums += local_quadform_sums;
        }
    }

    ArrayXd log_densities = -0.5 * (C - log_det + quadform_sums);

    return log_densities;
}

// Function to generate random samples from a Matérn-like field using eigendecomposition
// [[Rcpp::export]]
Eigen::MatrixXd rmatern_eigen(int n, int dim_x, int dim_y, double rho1, double rho2, int nu) {
    // Step 1: Create 1D AR(1) precision matrices
    SparseMatrix<double> Q1 = make_AR_prec_matrix(dim_x, rho1);
    SparseMatrix<double> Q2 = make_AR_prec_matrix(dim_y, rho2);
    
    // Step 2: Perform eigendecomposition of Q1 and Q2
    SelfAdjointEigenSolver<SparseMatrix<double>> solver1(Q1);
    VectorXd A1 = solver1.eigenvalues();
    MatrixXd V1 = solver1.eigenvectors();

    SelfAdjointEigenSolver<SparseMatrix<double>> solver2(Q2);
    VectorXd A2 = solver2.eigenvalues();
    MatrixXd V2 = solver2.eigenvectors();

    // Step 3: Prepare for sampling
    int D = dim_x * dim_y;
    MatrixXd samples = MatrixXd::Zero(D, n);

    // Random number generation
    std::vector<std::mt19937> generators(omp_get_max_threads());
    for (auto& gen : generators) {
        std::random_device rd;
        gen.seed(rd());
    }
    std::normal_distribution<> d(0, 1);

    // Step 5: Generate samples
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        MatrixXd local_samples = MatrixXd::Zero(D, n);
        VectorXd v(D);

        #pragma omp for collapse(2) nowait
        for (int i = 0; i < dim_x; ++i) {
            for (int j = 0; j < dim_y; ++j) {
                double lambda = std::pow(A1(i) + A2(j), -(nu + 1.0) / 2.0);
                v = kroneckerProduct(V1.col(i), V2.col(j));
                
                for (int s = 0; s < n; ++s) {
                    local_samples.col(s) += lambda * d(generators[thread_id]) * v;
                }
            }
        }

        #pragma omp critical
        {
            samples += local_samples;
        }
    }

    return samples;
}