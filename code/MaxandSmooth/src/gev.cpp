#include "gev.h"

#include <RcppEigen.h>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <nlopt.hpp>
#include <omp.h>
#include <cmath>
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(nloptr)]]

using namespace Rcpp;
using namespace autodiff;
using namespace Eigen;


namespace gev {

/**
 * @brief Link function for GEV location parameter (mu)
 */
dual mu_link(const dual& psi) {
    return exp(psi);
}

/**
 * @brief Link function for GEV scale parameter (sigma)
 */
dual sigma_link(const dual& psi, const dual& tau) {
    return exp(psi + tau);
}

/**
 * @brief Link function for GEV shape parameter (xi)
 */
dual xi_link(const dual& phi) {
    const double c_phi = 0.8;
    const double b_phi = 0.39563;
    const double a_phi = 0.062376;

    dual xi = (phi - a_phi) / b_phi;
    xi = 1 - exp(-exp(xi));
    xi = pow(xi, 1.0/c_phi) - 0.5;

    return xi;
}



/**
 * @brief Computes the log-likelihood of the GEV distribution.
 *
 * @param params Transformed GEV parameters (psi, tau, phi).
 * @param data Observed data points.
 * @return The computed log-likelihood.
 */
dual loglik(dual psi, dual tau, dual phi, const VectorXd& data) {
    try {
        dual mu = mu_link(psi);
        dual sigma = sigma_link(psi, tau);
        dual xi = xi_link(phi);
        
        if (val(sigma) < 1e-10) {
            return -1e10;  // Return large negative value for invalid sigma
        }
        
        int n = data.size();
        dual loglik = 0;
        loglik -= 0.5 * pow(psi / 10, 2);
        loglik -= 0.5 * pow(tau / 4, 2);
        loglik -= 0.5 * pow(phi / 2, 2);
        
        for(int i = 0; i < n; ++i) {
            dual z = (data(i) - mu) / sigma;
            
            if (abs(xi) < 1e-6) {
                dual term = log(sigma) + z + exp(-z);
                if (std::isfinite(val(term))) {
                    loglik -= term;
                } else {
                    return -1e10;
                }
            } else {
                dual t = 1 + xi * z;
                if (val(t) <= 0) {
                    return -1e10;
                }
                dual term = log(sigma) + (1.0 + 1.0 / xi) * log(t) + pow(t, -1.0 / xi);
                if (std::isfinite(val(term))) {
                    loglik -= term;
                } else {
                    return -1e10;
                }
            }
        }
        
        return loglik;
    } catch (...) {
        return -1e10;
    }
}

/**
 * @brief Evaluates the log-likelihood value for given GEV parameters and data.
 *
 * @param params The transformed GEV parameters (psi, tau, phi).
 * @param data The observed data points.
 * @return The log-likelihood value as a double.
 */
double loglik_value(dual psi, dual tau, dual phi, const Eigen::VectorXd& data) {
    auto f = [&data](dual psi, dual tau, dual phi) { return loglik(psi, tau, phi, data); };
    return val(f(psi, tau, phi));
}

/**
 * @brief Objective function for NLopt optimization.
 *
 * @param n Number of parameters.
 * @param x Pointer to the parameter array.
 * @param grad Pointer to the gradient array.
 * @param f_data Pointer to additional data (Eigen::VectorXd).
 * @return The objective function value as a double.
 */
double objective(unsigned n, const double* x, double* grad, void* f_data) {
    const Eigen::VectorXd* pData = reinterpret_cast<const Eigen::VectorXd*>(f_data);
    const Eigen::VectorXd& dataVec = *pData;
    dual psi = x[0];
    dual tau = x[1];
    dual phi = x[2];

    double loglik_val = loglik_value(psi, tau, phi, dataVec);

    if (grad) {
        VectorXdual grads;
        auto f = [&dataVec](dual psi, dual tau, dual phi) -> dual { 
            return loglik(psi, tau, phi, dataVec); 
        };
        grads = gradient(f, wrt(psi, tau, phi), at(psi, tau, phi));
        for (int i = 0; i < 3; ++i) {
            grad[i] = -val(grads(i));
        }
    }

    return -loglik_val;  // Negative because we're minimizing
}

/**
 * @brief Performs maximum likelihood estimation for GEV parameters.
 *
 * @param data The observed data points.
 * @return The estimated GEV parameters as Eigen::Vector3d.
 */
Eigen::Vector3d mle(const Eigen::VectorXd& data) {
    try {
        // Check input data
        if (data.size() == 0) {
            throw std::runtime_error("Empty data vector");
        }
        
        nlopt::opt opt(nlopt::LD_LBFGS, 3);  // 3 parameters
        
        // More conservative initial parameters
        Eigen::Vector3d initial_params;
        
        initial_params << std::log(5.0), 
                         std::log(5.0) - std::log(1.0), 
                         0.01;
        
        opt.set_min_objective(objective, (void*)&data);
        opt.set_ftol_rel(1e-6);
        opt.set_xtol_rel(1e-6);
        opt.set_maxeval(1000);
        
        // Add bounds to prevent numerical issues
        std::vector<double> lb(3, -10.0);
        std::vector<double> ub(3, 10.0);
        opt.set_lower_bounds(lb);
        opt.set_upper_bounds(ub);
        
        std::vector<double> x(initial_params.data(), initial_params.data() + initial_params.size());
        double minf;
        
        nlopt::result result = opt.optimize(x, minf);
        
        if (result < 0) {
            throw std::runtime_error("NLopt optimization failed with code " + std::to_string(result));
        }
        
        return Eigen::Vector3d(x[0], x[1], x[2]);
    } catch (std::exception& e) {
        throw std::runtime_error(std::string("Error in mle: ") + e.what());
    }
}

/**
 * @brief Link function for GEV location parameter (mu)
 */
dual2nd mu_link_2nd(const dual2nd& psi) {
    return exp(psi);
}

/**
 * @brief Link function for GEV scale parameter (sigma)
 */
dual2nd sigma_link_2nd(const dual2nd& psi, const dual2nd& tau) {
    return exp(psi + tau);
}

/**
 * @brief Link function for GEV shape parameter (xi)
 */
dual2nd xi_link_2nd(const dual2nd& phi) {
    const double c_phi = 0.8;
    const double b_phi = 0.39563;
    const double a_phi = 0.062376;

    dual2nd xi = (phi - a_phi) / b_phi;
    xi = 1 - exp(-exp(xi));
    xi = pow(xi, 1.0/c_phi) - 0.5;

    return xi;
}

/**
 * @brief Alternate version of the log-likelihood where the parameters are twice differentiable.
 *
 * @param params The transformed GEV parameters (psi, tau, phi).
 * @param data The observed data points.
 * @return The computed log-likelihood.
 */
dual2nd loglik_2nd(dual2nd psi, dual2nd tau, dual2nd phi, const VectorXd& data) {
    dual2nd mu = mu_link_2nd(psi);
    dual2nd sigma = sigma_link_2nd(psi, tau);
    dual2nd xi = xi_link_2nd(phi);

    int n = data.size();
    dual2nd loglik = 0;

    for(int i = 0; i < n; ++i) {
        dual2nd z = (data(i) - mu) / sigma;

        if (abs(xi) < 1e-6) {
            loglik -= log(sigma) + z + exp(-z);
        } else {
            dual2nd t = 1 + xi * z;
            loglik -= log(sigma) + (1.0 + 1.0 / xi) * log(t) + pow(t, -1.0 / xi);
        }
    }

    return loglik;
}

/**
 * @brief Computes the Hessian of the log-likelihood of the GEV distribution.
 *
 * @param params The transformed GEV parameters (psi, tau, phi).
 * @param data The observed data points.
 * @return The computed Hessian matrix.
 */
Eigen::MatrixXd loglik_hessian(dual2nd psi, dual2nd tau, dual2nd phi, const Eigen::VectorXd& data) {

    // Define a lambda function that takes dual parameters and returns the log-likelihood
    auto f = [&data](dual2nd psi, dual2nd tau, dual2nd phi) -> dual2nd {
        return loglik_2nd(psi, tau, phi, data);
    };

    // Compute the Hessian matrix using autodiff's hessian function
    Eigen::Matrix<dual2nd, Eigen::Dynamic, Eigen::Dynamic> hess_dual = autodiff::hessian(f, wrt(psi, tau, phi), at(psi, tau, phi));

    // Convert the Hessian from dual to double precision
    Eigen::MatrixXd hess = hess_dual.cast<double>();

    return hess;
}

/**
 * @brief Performs maximum likelihood estimation for GEV parameters for multiple locations in parallel.
 *
 * @param data A matrix where each column represents data for a location.
 * @return A list containing the MLEs and Hessians for each location.
 */
Rcpp::List mle_multiple(Eigen::MatrixXd& data) {
    // Input validation
    if (data.cols() == 0 || data.rows() == 0) {
        Rcpp::stop("Input data matrix is empty");
    }

    int n_locations = data.cols();
    Eigen::MatrixXd results(n_locations, 3);
    Eigen::MatrixXd hessians(n_locations, 9);
    
    #pragma omp parallel for
    for (int i = 0; i < n_locations; ++i) {
        try {
            // Check for invalid data
            if (data.col(i).hasNaN() || !data.col(i).allFinite()) {
                throw std::runtime_error("Invalid data detected");
            }

            Eigen::Vector3d mle_result = mle(data.col(i));
            
            // Check for invalid MLE results
            if (!mle_result.allFinite()) {
                throw std::runtime_error("MLE computation failed");
            }

            dual2nd psi = mle_result(0);
            dual2nd tau = mle_result(1);
            dual2nd phi = mle_result(2);
            
            Eigen::MatrixXd hess = loglik_hessian(psi, tau, phi, data.col(i));
            
            // Check for invalid Hessian
            if (!hess.allFinite()) {
                throw std::runtime_error("Hessian computation failed");
            }

            results.row(i) = mle_result;
            hessians.row(i) = Eigen::Map<const Eigen::VectorXd>(hess.data(), hess.size());
        }
        catch (const std::exception& e) {
            #pragma omp critical
            {
                Rcpp::stop("Error at location " + std::to_string(i) + ": " + e.what());
            }
        }
    }

    return Rcpp::List::create(
        Rcpp::Named("mles") = results,
        Rcpp::Named("hessians") = hessians
    );
}

} // namespace gev