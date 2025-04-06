# Matérn-like Copula in Max-and-Smooth

## 1. Statistical Framework

### 1.1 Model Specification

Let $Y$ be an $n \times p$ matrix of observations where:
- Rows $(i=1,\ldots,n)$ represent temporal replicates
- Columns $(j=1,\ldots,p)$ represent spatial locations

The model combines:

1. **GEV Marginals**:
   $$Y_{ij} \sim \text{GEV}(\mu_j, \sigma_j, \xi_j)$$
   
   with parameter transformations:
   - $\mu_j = \exp(\psi_j)$ for positivity
   - $\sigma_j = \exp(\psi_j + \tau_j)$ for positivity and scale dependence
   - $\xi_j = f(\phi_j)$ where $f(\phi)$ is a carefully chosen function

2. **Gaussian Copula Transform**:
   $$Z_{ij} = \Phi^{-1}(F_{\text{GEV}}(Y_{ij}|\mu_j,\sigma_j,\xi_j))$$

3. **Matérn-like Precision Structure**:
   $$Z_t \sim \mathcal{N}(0, Q^{-1}), \quad Q = (Q_{\rho_1} \otimes I_{n_2} + I_{n_1} \otimes Q_{\rho_2})^{\nu+1}$$

where:
- $Q_{\rho}$ is the precision matrix of a standardized AR(1) process
- $\otimes$ denotes the Kronecker product
- $\nu$ is a smoothness parameter
- $Q$ has been scaled to have unit variance

### 1.2 Likelihood Structure

The total log-likelihood combines marginal and copula components:

$$\ell(\theta|Y) = \sum_{j=1}^p \sum_{i=1}^n \ell_{\text{GEV}}(y_{ij}|\mu_j,\sigma_j,\xi_j) + \ell_{\text{copula}}(Z|Q)$$

where the copula log-likelihood is:

$$\ell_{\text{copula}}(Z|Q) = \frac{1}{2}\log|Q| - \frac{1}{2}Z^TQZ + \frac{1}{2}Z^TZ$$

The Matérn-like structure ensures:

- A wide range of possible spatial correlation structures
- Computational efficiency via sparse precision matrices
- Interpretable correlation/range parameters $\rho_1, \rho_2$
- Controllable smoothness via $\nu$

## 2. Computational Strategy

### 2.1 Algorithm Overview

The algorithm proceeds in four distinct steps:

1. **Initial Max Step (IID)**:
   - Fit independent GEV distributions to each location
   - Obtain initial estimates for $(\psi_j, \tau_j, \phi_j)$
   - Use parameter transformations to ensure valid ranges
   - Can be done in parallel for each location because of independence of the GEV marginals
   
2. **Copula Parameter Estimation**:
   - Transform data to empirical quantiles
   - Convert to standard normal margins
   - Optimize Matérn parameters $(\rho_1, \rho_2)$ via ML
   
3. **Max Step with Copula**:
   - Re-estimate GEV parameters with spatial dependence
   - Use IID estimates as starting values
   - Assume that the copula parameters are known
   - Combine marginal and copula likelihoods
   - Include weak parameter priors for regularization
   
4. **Smooth Step**:
   - Input: MLEs $\hat{\eta}$ and Hessian-based precision $Q_{\eta y}$
   - Perform spatial smoothing of the MLEs using a BYM2 model
   - For each parameter type $k \in \{\psi, \tau, \phi\}$:
     $$\eta_k = \mu_k\mathbf{1} + \sigma_k(\sqrt{\rho_k/c}\eta^{\mathrm{spatial}}_k + \sqrt{1-\rho_k}\eta^{\mathrm{random}}_k)$$
   - Implementation via Stan's efficient HMC sampler
   - Custom multivariate normal likelihood based on sparse Cholesky decomposition of $Q_{\eta y}$

### 2.2 Implementation Details

1. **Efficiency**:
   - Parallel processing for location-wise computations
      - Fast parallelisation of the IID step
   - Sparse matrix operations for spatial precision
      - Large-scale spatial Copula estimation is possible because of the Kronecker structure
   - Automatic differentiation for gradients
      - Easy to add new data-level likelihoods
   - Stan's HMC sampler is very efficient when everything is Gaussian

2. **Robustness**:
   - Parameter transformations ensure valid ranges
   - Careful handling of numerical issues in GEV likelihood
   - Proper uncertainty propagation between most steps
      - Uncertainty in the copula parameters is not propagated to final result
   - Regularizing priors for stability
   - Stan provides informative MCMC diagnostics

3. **Flexibility**:
   - Modular design allows different marginal distributions
   - Adjustable spatial dependence through copula parameters
   - Customizable prior specifications
   - Extensible to other spatial models

## 4. Software Implementation

The implementation is provided in the `maxandsmooth` R package (currently under development):

1. **Core Components**:
   - Efficient C++ implementation using Rcpp/RcppEigen
   - Automatic differentiation for gradients
   - Parallel processing via OpenMP
   - Stan integration for HMC sampling

2. **Key Features**:
   - User-friendly R interface
   - Flexible model specification
   - Comprehensive uncertainty quantification
   - Extensive diagnostic tools

3. **Performance**:
   - Scales to large spatial datasets
   - Efficient memory usage
   - Robust to different data scenarios

## 5. Example Application

The method is demonstrated on the UKCP precipitation projections:

1. **Data Description**:
   - Yearly maximum hourly rainfall data
   - Spatial coverage of UK (44,000 grid cells)
   - Multiple decades of projections

2. **Results**:
   - Improved parameter estimates vs IID approach
   - Coherent spatial patterns in GEV parameters
   - Proper uncertainty quantification
   - Computational efficiency demonstrated

3. **Comparisons**:
   - Better than independent fits
   - Comparable to full Bayesian inference
   - Significantly faster computation
   - More stable parameter estimates

This stepwise approach provides a computationally efficient method for spatial extreme value analysis while maintaining proper uncertainty quantification throughout the inference pipeline.