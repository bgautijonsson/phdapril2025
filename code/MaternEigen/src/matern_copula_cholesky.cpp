#include <RcppEigen.h>
#include <omp.h>
#include <random>
#include "ar_matrix.h"


// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(openmp)]]

using namespace Rcpp;
using namespace Eigen;

// Helper function to compute D
// [[Rcpp::export]]
Eigen::VectorXd marginal_sd_cholesky(const Eigen::SparseMatrix<double>& L, int nu) {
    int N = L.rows();
    Eigen::MatrixXd L_inv = L.triangularView<Eigen::Lower>().solve(Eigen::MatrixXd::Identity(N, N));
    
    Eigen::MatrixXd temp = L_inv;
    for (int i = 0; i < nu; ++i) {
        if (i % 2 == 0) {
            temp = L_inv.transpose() * temp;
        } else {
            temp = L_inv * temp;
        }
    }
    
    return temp.colwise().squaredNorm().cwiseSqrt();
}


// [[Rcpp::export]]
Eigen::VectorXd dmatern_copula_cholesky(const Eigen::MatrixXd& X, int dim_x, int dim_y, double rho1, double rho2, int nu) {
    int n_obs = X.cols();
    int N = dim_x * dim_y;

    // Create base precision matrix Q
    Eigen::SparseMatrix<double> Q1 = make_AR_prec_matrix(dim_x, rho1);
    Eigen::SparseMatrix<double> Q2 = make_AR_prec_matrix(dim_y, rho2);
    Eigen::SparseMatrix<double> I1 = Eigen::SparseMatrix<double>(dim_x, dim_x);
    Eigen::SparseMatrix<double> I2 = Eigen::SparseMatrix<double>(dim_y, dim_y);
    I1.setIdentity();
    I2.setIdentity();
    Eigen::SparseMatrix<double> Q = Eigen::kroneckerProduct(Q1, I2) + Eigen::kroneckerProduct(I1, Q2);

    // Compute Cholesky decomposition
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Lower, Eigen::NaturalOrdering<int>> cholSolver(Q);
    
    if (cholSolver.info() != Eigen::Success) {
        throw std::runtime_error("Cholesky decomposition failed");
    }

    // Get the sparse Cholesky factor L
    Eigen::SparseMatrix<double> L = cholSolver.matrixL();
    Eigen::VectorXd D = marginal_sd_cholesky(L, nu);

    // Compute log-determinant
    double log_det = D.log().sum() + (nu + 1) * L.diagonal().array().log().sum();
    // Compute quadratic forms
    Eigen::VectorXd quadform_sums(n_obs);
    for (int i = 0; i < n_obs; ++i) {
        Eigen::VectorXd y = D.cwiseProduct(X.col(i));
        for (int j = 0; j <= nu; ++j) {
            if (j % 2 == 0) {
                y = L.transpose() * y;
            } else {
                y = L * y;
            }
        }
        quadform_sums(i) = y.squaredNorm();
    }

    // Subtract x'Ix (sum of squared elements for each observation)
    Eigen::VectorXd x_squared = X.colwise().squaredNorm();

    // Compute log densities
    Eigen::VectorXd log_densities = log_det - 0.5 * (quadform_sums.array() - x_squared.array());

    return log_densities;
}