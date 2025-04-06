# The Max-and-Smooth Method: A Two-Step Approach for Approximate Bayesian Inference in Latent Gaussian Models

## 1. Introduction

Max-and-Smooth is an approximate Bayesian inference method designed for extended Latent Gaussian Models (LGMs) with independent data replicates. The method combines efficient maximum likelihood estimation with Gaussian approximations to provide fast and accurate posterior inference, particularly well-suited for high-dimensional problems.

## 2. Model Structure

### 2.1 Extended Latent Gaussian Models

An extended LGM consists of three hierarchical levels:

1. **Data Level**: For observations grouped into $G$ groups with $n_i$ observations per group $i$, the data density is:

   $$p(y|\psi_1,\ldots,\psi_M) = \prod_{i=1}^G \prod_{j \in A_i} p(y_{ij}|\psi_{1,i},\psi_{2,i},\ldots,\psi_{M,i})$$

   where $A_i$ is the index set for group $i$ and $\psi_{m,i}$ are parameters.

2. **Latent Level**: Parameters are transformed through an $M$-variate link function:

   $$g: D \to \mathbb{R}^M$$
   $$g(\psi_{1,i},\psi_{2,i},\ldots,\psi_{M,i}) = (\eta_{1,i},\eta_{2,i},\ldots,\eta_{M,i})^T$$

   The transformed parameters follow linear models:
   
   $$\eta_m = X_m\beta_m + A_mu_m + \epsilon_m, \quad m=1,\ldots,M$$

   where:
   - $\beta_m$ are fixed effects
   - $u_m$ are random effects
   - $\epsilon_m$ are model errors
   - $X_m$ and $A_m$ are design matrices

3. **Hyperparameter Level**: Prior distributions for the hyperparameters $\theta$ controlling the random effects.

### 2.2 Prior Structure

The prior distributions at the latent level are Gaussian:

- $\beta_m \sim N(\mu_{\beta,m}, Q_{\beta,m}^{-1})$
- $u_m \sim N(0, Q_{u,m}^{-1}(\theta))$
- $\epsilon_m \sim N(0, Q_{\epsilon,m}^{-1})$

where $Q$ matrices are precision matrices (inverse covariance matrices).

## 3. The Max-and-Smooth Algorithm

### 3.1 Step 1: Max - Gaussian Approximation

The first step approximates the likelihood function with a Gaussian density. Two approaches are available:

#### 3.1.1 ML-Based Approximation

For each group $i$:

1. Compute the Maximum Likelihood Estimate (MLE):
   $$\hat{\eta}_i = \arg\max_{\eta_i} L(\eta_i|y_i)$$

2. Compute the observed information matrix:
   $$I_{\eta y,i} = -\nabla^2 \log L(\eta_i|y_i)|_{\eta_i=\hat{\eta}_i}$$

3. Approximate the likelihood:
   $$L(\eta_i|y_i) \approx c_i\hat{L}(\eta_i|y_i) = c_iN(\eta_i|\hat{\eta}_i, I_{\eta y,i}^{-1})$$

#### 3.1.2 Normalized Likelihood Approximation

Alternatively:

1. Normalize the likelihood to obtain a proper density:
   $$L_N(\eta_i|y_i) = \frac{L(\eta_i|y_i)}{\int L(\eta_i|y_i)d\eta_i}$$

2. Compute mean and covariance:
   $$\tilde{\eta}_i = E_{L_N}[\eta_i], \quad \Omega_{\eta y,i} = \text{Var}_{L_N}[\eta_i]$$

3. Approximate the likelihood:
   $$L(\eta_i|y_i) \approx d_i\tilde{L}(\eta_i|y_i) = d_iN(\eta_i|\tilde{\eta}_i, \Omega_{\eta y,i})$$

### 3.2 Step 2: Smooth - Gaussian-Gaussian Inference

The second step performs Bayesian inference using the approximated likelihood. The resulting posterior has the form:

$$p(\eta,\nu,\theta|y) \propto p(\theta)p(\eta,\nu|\theta)\prod_{i=1}^G N(\eta_i|\hat{\eta}_i, \Sigma_{\eta y,i})$$

where $\nu$ contains all latent parameters except $\eta$.

The inference proceeds by:

1. Sampling from the marginal posterior of $\theta$:
   $$p(\theta|y) \propto p(\theta)\int p(\eta,\nu|\theta)\prod_{i=1}^G N(\eta_i|\hat{\eta}_i, \Sigma_{\eta y,i})d\eta d\nu$$

2. For each $\theta$ sample, drawing from the conditional posterior $p(\eta,\nu|\theta,y)$, which is Gaussian.

## 4. Computational Efficiency

The method is particularly efficient because:

1. The approximate posterior has a Gaussian-Gaussian structure, enabling fast sampling.
2. The marginal posterior of hyperparameters can be computed analytically.
3. The computational cost is nearly independent of the number of data replicates.
4. When using sparse precision matrices, computation scales well with latent dimension.

## 5. Accuracy and Convergence

The accuracy of Max-and-Smooth depends on:

1. The quality of the Gaussian approximation to the likelihood
2. The number of independent replicates per parameter
3. The complexity of the latent structure

The method becomes increasingly accurate as the number of replicates increases, due to the asymptotic normality of the likelihood function.

## 6. Implementation Considerations

For efficient implementation:

1. Use sparse matrix operations when possible
2. Parallelize the computation of MLEs across groups
3. Employ efficient sampling methods for the hyperparameters
4. Consider numerical stability in the computation of log-determinants

## References

1. Hrafnkelsson, B., Siegert, S., Huser, R., Bakka, H., & Jóhannesson, Á. V. (2021). Max-and-Smooth: a two-step approach for approximate Bayesian inference in latent Gaussian models. Bayesian Analysis, 16(2), 611-638.

2. Rue, H., Martino, S., & Chopin, N. (2009). Approximate Bayesian inference for latent Gaussian models by using integrated nested Laplace approximations. Journal of the Royal Statistical Society: Series B, 71(2), 319-392.
