#include "circulant_utils.h"
#include <RcppEigen.h>
#include <complex>
#include <random>
#include <fftw3.h>

// [[Rcpp::depends(RcppEigen)]]

using namespace Rcpp;
using namespace Eigen;

Eigen::MatrixXcd fft2(const Eigen::MatrixXcd& input, double scale) {
    int dim2 = input.rows();
    int dim1 = input.cols();
    Eigen::MatrixXcd output(dim2, dim1);

    fftw_complex *in, *out;
    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * dim1 * dim2);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * dim1 * dim2);

    // Copy data from Eigen matrix to FFTW input
    for (int i = 0; i < dim2; ++i) {
        for (int j = 0; j < dim1; ++j) {
            in[i*dim1 + j][0] = input(i, j).real();
            in[i*dim1 + j][1] = input(i, j).imag();
        }
    }

    // Forward FFT
    fftw_plan plan = fftw_plan_dft_2d(dim2, dim1, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);

    // Copy result back to Eigen matrix and scale
    for (int i = 0; i < dim2; ++i) {
        for (int j = 0; j < dim1; ++j) {
            output(i, j) = std::complex<double>(out[i*dim1 + j][0], out[i*dim1 + j][1]) * scale;
        }
    }

    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);

    return output;
}

// Function to create the base matrix c
// [[Rcpp::export]]
Eigen::MatrixXd create_base_matrix(int dim1, int dim2, double rho1, double rho2) {
    Eigen::MatrixXd c = Eigen::MatrixXd::Zero(dim2, dim1);
    Eigen::VectorXd c1 = Eigen::VectorXd::Zero(dim1);
    Eigen::VectorXd c2 = Eigen::VectorXd::Zero(dim2);

    double scale1 = 1.0 / (1.0 - rho1 * rho1);
    double scale2 = 1.0 / (1.0 - rho2 * rho2);

    c1[0] = 1 + rho1 * rho1;
    c1[1] = -rho1;
    c1[dim1 - 1] = -rho1;
    c1 *= scale1;

    c2[0] = 1 + rho2 * rho2;
    c2[1] = -rho2;
    c2[dim2 - 1] = -rho2;
    c2 *= scale2;
    
    // Set the first row
    c(0, 0) = c1[0] + c2[0];
    c(0, 1) = c1[1];
    c(0, dim1 - 1) = c1[dim1 - 1];
    
    // Set the second and last row
    c(1, 0) = c2[1];
    c(dim2 - 1, 0) = c2[dim2 - 1];
    
    return c;
}

// Function to compute FFT of c, apply nu, and rescale eigenvalues using marginal variance
// [[Rcpp::export]]
Eigen::MatrixXcd compute_and_rescale_eigenvalues(const Eigen::MatrixXd& c, int nu) {
    int dim2 = c.rows();
    int dim1 = c.cols();

    fftw_complex *in, *out;
    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * dim1 * dim2);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * dim1 * dim2);

    // Copy data from Eigen matrix to FFTW input
    for (int i = 0; i < dim2; ++i) {
        for (int j = 0; j < dim1; ++j) {
            in[i*dim1 + j][0] = c(i, j);
            in[i*dim1 + j][1] = 0.0;
        }
    }

    // Forward FFT
    fftw_plan plan_forward = fftw_plan_dft_2d(dim2, dim1, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan_forward);

    // Copy eigenvalues to Eigen matrix and apply nu
    Eigen::MatrixXcd eigenvalues(dim2, dim1);
    for (int i = 0; i < dim2; ++i) {
        for (int j = 0; j < dim1; ++j) {
            std::complex<double> eig(out[i*dim1 + j][0], out[i*dim1 + j][1]);
            eigenvalues(i, j) = std::pow(eig, nu + 1);
        }
    }

    // Compute inverse of eigenvalues
    Eigen::MatrixXcd inv_eigenvalues = eigenvalues.cwiseInverse();

    // Inverse FFT of inverse eigenvalues
    for (int i = 0; i < dim2 * dim1; ++i) {
        in[i][0] = inv_eigenvalues(i / dim1, i % dim1).real();
        in[i][1] = inv_eigenvalues(i / dim1, i % dim1).imag();
    }

    fftw_plan plan_backward = fftw_plan_dft_2d(dim2, dim1, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(plan_backward);

    // Get marginal variance (first element of inverse FFT result)
    double mvar = out[0][0] / (dim1 * dim2);

    // Scale eigenvalues
    Eigen::MatrixXcd updated_eigenvalues = mvar * eigenvalues;

    fftw_destroy_plan(plan_forward);
    fftw_destroy_plan(plan_backward);
    fftw_free(in);
    fftw_free(out);

    return updated_eigenvalues;
}

// Function for matrix-vector product using FFT and eigenvalues
// [[Rcpp::export]]
Eigen::VectorXd matrix_vector_product(const Eigen::MatrixXcd& eigenvalues, const Eigen::VectorXd& v) {
    int dim2 = eigenvalues.rows();
    int dim1 = eigenvalues.cols();

    fftw_complex *in, *out;
    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * dim1 * dim2);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * dim1 * dim2);
    
    // Copy data from Eigen vector to FFTW input
    for (int i = 0; i < dim1 * dim2; ++i) {
        in[i][0] = v(i);
        in[i][1] = 0.0;
    }

    // Forward FFT
    fftw_plan plan_forward = fftw_plan_dft_2d(dim2, dim1, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan_forward);

    // Multiply with eigenvalues in frequency domain
    for (int i = 0; i < dim2; ++i) {
        for (int j = 0; j < dim1; ++j) {
            std::complex<double> temp(out[i*dim1 + j][0], out[i*dim1 + j][1]);
            temp *= eigenvalues(i, j);
            out[i*dim1 + j][0] = temp.real();
            out[i*dim1 + j][1] = temp.imag();
        }
    }

    // Inverse FFT
    fftw_plan plan_backward = fftw_plan_dft_2d(dim2, dim1, out, in, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(plan_backward);

    // Copy result back to Eigen vector and normalize
    Eigen::VectorXd result(dim1 * dim2);
    for (int i = 0; i < dim1 * dim2; ++i) {
        result(i) = in[i][0] / (dim1 * dim2);
    }

    fftw_destroy_plan(plan_forward);
    fftw_destroy_plan(plan_backward);
    fftw_free(in);
    fftw_free(out);

    return result;
}