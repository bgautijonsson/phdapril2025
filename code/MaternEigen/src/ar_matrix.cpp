#include "ar_matrix.h"
#include <RcppEigen.h>

// [[Rcpp::depends(RcppEigen)]]

// Function to create a 1-dimensional AR(1) precision matrix
// [[Rcpp::export]]
Eigen::SparseMatrix<double> make_AR_prec_matrix(int dim, double rho) {
    double scaling = 1.0 / (1.0 - rho * rho);
    double off_diags = -rho * scaling;
    double diag = (1.0 + rho * rho) * scaling;

    Eigen::SparseMatrix<double> Q(dim, dim);
    Q.reserve(Eigen::VectorXi::Constant(dim, 3));  // Reserve space for 3 non-zeros per column

    for (int i = 0; i < dim; ++i) {
        Q.insert(i, i) = (i == 0 || i == dim - 1) ? scaling : diag;
        if (i > 0) Q.insert(i, i-1) = off_diags;
        if (i < dim - 1) Q.insert(i, i+1) = off_diags;
    }

    Q.makeCompressed();
    return Q;
}