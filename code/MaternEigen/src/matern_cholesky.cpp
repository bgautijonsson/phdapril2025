#include <RcppEigen.h>
#include <omp.h>
#include <random>
#include "ar_matrix.h"


// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(openmp)]]

using namespace Rcpp;
using namespace Eigen;

// Function to create the standardized Matérn precision matrix with eigendecomposition method
// [[Rcpp::export]]
Eigen::SparseMatrix<double> make_matern_prec_matrix(int dim_x, int dim_y, double rho1, double rho2, int nu) {
  // Create precision matrices
  SparseMatrix<double> Q1 = make_AR_prec_matrix(dim_x, rho1);
  SparseMatrix<double> Q2 = make_AR_prec_matrix(dim_y, rho2);

  // Create full Q matrix using Kronecker sum
  SparseMatrix<double> I1(dim_x, dim_x);
  I1.setIdentity();

  SparseMatrix<double> I2(dim_y, dim_y);
  I2.setIdentity();


  SparseMatrix<double> Q = kroneckerProduct(Q1, I2) + kroneckerProduct(I1, Q2);

  // Apply matrix multiplication nu times
  if (nu > 0) {
    SparseMatrix<double> temp = Q;
    for (int i = 0; i < nu; ++i) {
      Q = Q * temp;
      Q.makeCompressed();
    }
  }


  return Q;
}

// [[Rcpp::export]]
Eigen::VectorXd dmatern_cholesky(const Eigen::MatrixXd& X, int dim_x, int dim_y, double rho1, double rho2, int nu) {
  int n_obs = X.cols();
  int N = dim_x * dim_y;
  const double C = N * std::log(2 * M_PI);

  // Create base precision matrix Q
  SparseMatrix<double> Q1 = make_AR_prec_matrix(dim_x, rho1);
  SparseMatrix<double> Q2 = make_AR_prec_matrix(dim_y, rho2);
  SparseMatrix<double> I1 = SparseMatrix<double>(dim_x, dim_x);
  SparseMatrix<double> I2 = SparseMatrix<double>(dim_y, dim_y);
  I1.setIdentity();
  I2.setIdentity();
  SparseMatrix<double> Q = kroneckerProduct(Q1, I2) + kroneckerProduct(I1, Q2);

  // Compute Cholesky decomposition
  Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Lower, Eigen::NaturalOrdering<int>> cholSolver(Q);
  
  if (cholSolver.info() != Eigen::Success) {
    throw std::runtime_error("Cholesky decomposition failed");
  }

  // Get the sparse Cholesky factor L
  SparseMatrix<double> L = cholSolver.matrixL();

  // Compute log-determinant
  double log_det = 2 * (nu + 1) * L.diagonal().array().log().sum();

  // Compute quadratic forms
  VectorXd quadform_sums(n_obs);
  #pragma omp parallel for
  for (int i = 0; i < n_obs; ++i) {
    VectorXd y = X.col(i);
    for (int j = 0; j <= nu; ++j) {
      if (j % 2 == 0) {
        y = L.transpose() * y;
      } else {
        y = L * y;
      }
    }
    quadform_sums(i) = y.squaredNorm();
  }

  // Compute log densities
  VectorXd log_densities = -0.5 * (C - log_det + quadform_sums.array());

  return log_densities;
}