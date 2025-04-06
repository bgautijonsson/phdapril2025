#ifndef CIRCULANT_UTILS
#define CIRCULANT_UTILS

// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

Eigen::MatrixXcd fft2(const Eigen::MatrixXcd& input, double scale);
Eigen::MatrixXd create_base_matrix(int dim1, int dim2, double rho1, double rho2);
Eigen::MatrixXcd compute_and_rescale_eigenvalues(const Eigen::MatrixXd& c, int nu);
Eigen::VectorXd matrix_vector_product(const Eigen::MatrixXcd& eigenvalues, const Eigen::VectorXd& v);


#endif