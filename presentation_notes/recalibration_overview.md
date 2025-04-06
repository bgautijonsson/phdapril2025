# Recalibration by Latent Modeling of UKCP Projections

In **Paper 2**, we propose a **hierarchical Bayesian** framework to recalibrate high-resolution climate simulations from the UKCP ensemble using both the **UK Met Office observed data** and the **UKCP model outputs** themselves. The overarching goal is to **improve** sub-daily extreme precipitation forecasts by explicitly modeling and correcting the **bias** in the simulation parameters.

In one way, this can be viewed as **recalibrating the UKCP ensemble** to the observed data, but in another way, it is a **latent variable model** where we use the simulated data to add information to the observed data.

---

## 1. Notation and Observations

1. Let $\mathbf{Y}^{\text{obs}} = \{Y_{\ell}^{\text{obs}}\}_{\ell \in \mathcal{L}}$ denote the **observed** sub-daily precipitation data at locations $\ell \in \mathcal{L}$.  
2. Let $\{\mathbf{Y}^{\text{sim}}_r\}_{r=1}^R$ denote **simulated** data from $R$ different UKCP runs (each run $r$ corresponds to the same spatial domain $\mathcal{L}$ but may differ due to initial conditions or climate model configurations).

---

## 2. Parameter Hierarchy

We suppose each location $\ell$ has **true physical parameters** $\boldsymbol{\theta}_{\ell}^{\text{true}}$ (e.g., for a GEV or other extremes distribution). Meanwhile, the **simulated** parameters from run $r$ at location $\ell$ are $\boldsymbol{\theta}_{\ell,r}^{\text{sim}}$. We posit a **hierarchical** relationship:

1. **True Parameter Layer**  
   $$
   \boldsymbol{\theta}_{\ell}^{\text{true}}
   \;\sim\;
   \pi\bigl(\boldsymbol{\theta}_{\ell}^{\text{true}} \mid \boldsymbol{\phi}\bigr),
   $$  
   where $\boldsymbol{\phi}$ are hyperparameters governing spatial structure (e.g., via Gaussian Markov random fields or BYM models).

2. **Simulator Parameter Layer**  
   For each UKCP run $r$, the parameters are assumed to deviate from the true parameters according to:
   $$
   \boldsymbol{\theta}_{\ell,r}^{\text{sim}}
   \;=\;
   \boldsymbol{\alpha}
   \;+\;
   \boldsymbol{\beta}\,\boldsymbol{\theta}_{\ell}^{\text{true}}
   \;+\;
   \boldsymbol{\varepsilon}_{\ell,r},
   $$
   where:
   - $\boldsymbol{\alpha}$ is an intercept (possible bias offset).  
   - $\boldsymbol{\beta}$ is a scaling matrix/factor capturing systematic bias in the simulator.  
   - $\boldsymbol{\varepsilon}_{\ell,r}$ represents residual noise, often modeled as 
     $\boldsymbol{\varepsilon}_{\ell,r} \sim \mathrm{Normal}\bigl(\mathbf{0}, \boldsymbol{\Sigma}_{\varepsilon}\bigr)$.

---

## 3. Observational and Simulator Data Models

We **link** these latent parameters to the actual observed data $\mathbf{Y}^{\text{obs}}$ and simulated data $\mathbf{Y}^{\text{sim}}_r$ via **data likelihoods**:

1. **Observed Data Likelihood**  
   $$
   Y_{\ell}^{\text{obs}}
   \;\sim\;
   f\bigl(\cdot \mid \boldsymbol{\theta}_{\ell}^{\text{true}}\bigr),
   $$
   where $f$ is, for example, the GEV distribution if we are modeling annual maxima, or another appropriate extremes distribution.

2. **Simulated Data Likelihood**  
   $$
   Y_{\ell,r}^{\text{sim}}
   \;\sim\;
   f\bigl(\cdot \mid \boldsymbol{\theta}_{\ell,r}^{\text{sim}}\bigr).
   $$
   Each UKCP run $r$ thus provides an independent realization from a distribution parameterized by $\boldsymbol{\theta}_{\ell,r}^{\text{sim}}$.

---

## 4. Full Hierarchical Model

Putting these components together yields a **joint posterior**:

$$
\begin{aligned}
p\bigl(\{\boldsymbol{\theta}_{\ell}^{\text{true}}\}, &\, \{\boldsymbol{\theta}_{\ell,r}^{\text{sim}}\}, \boldsymbol{\alpha}, \boldsymbol{\beta}, \boldsymbol{\phi}, \boldsymbol{\Sigma}_{\varepsilon} \;\bigm|\; \mathbf{Y}^{\text{obs}}, \{\mathbf{Y}^{\text{sim}}_r\}\bigr) \\
&\;\propto\;
\underbrace{\prod_{\ell \in \mathcal{L}} f\bigl(Y_{\ell}^{\text{obs}} \mid \boldsymbol{\theta}_{\ell}^{\text{true}}\bigr)}_{\text{Observed data likelihood}}
\times
\underbrace{\prod_{r=1}^R \prod_{\ell \in \mathcal{L}} f\bigl(Y_{\ell,r}^{\text{sim}} \mid \boldsymbol{\theta}_{\ell,r}^{\text{sim}}\bigr)}_{\text{Simulator data likelihoods}}
\\
&\quad\times
\underbrace{\prod_{r=1}^R p\bigl(\boldsymbol{\theta}_{\ell,r}^{\text{sim}} \mid \boldsymbol{\alpha}, \boldsymbol{\beta}, \boldsymbol{\theta}_{\ell}^{\text{true}}, \boldsymbol{\Sigma}_{\varepsilon}\bigr)}_{\text{Simulator-parameter link}}
\times
\underbrace{\prod_{\ell \in \mathcal{L}}\pi\bigl(\boldsymbol{\theta}_{\ell}^{\text{true}} \mid \boldsymbol{\phi}\bigr)}_{\text{Spatial prior}}
\times
\underbrace{\pi(\boldsymbol{\alpha}, \boldsymbol{\beta}, \boldsymbol{\phi}, \boldsymbol{\Sigma}_{\varepsilon})}_{\text{Hyperpriors}}.
\end{aligned}
$$

- **Spatial Prior** $\pi(\boldsymbol{\theta}_{\ell}^{\text{true}} \mid \boldsymbol{\phi})$ could encode Markov random field assumptions (e.g., BYM2 or ICAR) across locations $\ell$.  
- **Link Function** $\boldsymbol{\theta}_{\ell,r}^{\text{sim}} = \boldsymbol{\alpha} + \boldsymbol{\beta}\,\boldsymbol{\theta}_{\ell}^{\text{true}} + \boldsymbol{\varepsilon}_{\ell,r}$ enforces a linear bias-correction.  
- **Residual Covariance** $\boldsymbol{\Sigma}_{\varepsilon}$ captures run-to-run variability in the simulation biases.

---

## 5. Computational Strategy

The computational implementation combines efficient **copula-based dependence modeling** with a **two-step Max-and-Smooth approach**:

### 5.1 Max Step: Parameter Estimation

1. **Independent Estimates**:
   - First obtain independent GEV parameter estimates at each location
   - Uses parallel processing via C++ implementation
   
2. **Copula Integration**:
   - Add spatial dependence through Matérn-like Gaussian copula
   - Transform margins to standard normal
   
3. **Efficient Likelihood Computation**:
   - Combined likelihood includes both GEV margins and copula:
   $$
   \ell(\theta|Y) = \sum_{j=1}^N \left[\sum_{i=1}^P \ell_{\text{GEV}}(Y_{ij}|\mu_i,\sigma_i,\xi_i) + \frac{1}{2}\left(\log|\mathbf{Q}| - Z_j^T\mathbf{Q}Z_j + Z_j^TZ_j\right) \right]
   $$
   - Parallel processing of replicates
   - Sparse matrix operations for precision matrix $\mathbf{Q}$

### 5.2 Smooth Step: Spatial Modeling

1. **BYM2 Prior Structure**:
   - Implements spatial smoothing via BYM2 model
   - Proper scaling for interpretable spatial variance components
   
2. **Efficient MCMC Implementation**:
   - Uses Stan's HMC sampler
   - Custom functions for ICAR precision calculations
   - Sparse Cholesky decomposition for precision matrices

### 5.3 Computational Optimizations

1. **Precision Matrix Construction**:
   - Kronecker sum decomposition for Matérn-like structure
   - Efficient eigendecomposition for marginal variances
   
2. **Parallel Processing**:
   - OpenMP parallelization for likelihood computations
   - Efficient memory management for large grids
   - Thread-safe random number generation for simulation

3. **Memory Efficiency**:
   - Sparse matrix representations throughout
   - Careful management of Cholesky factors
   - Efficient storage of precision matrix components

This computational strategy enables efficient handling of large spatial datasets while properly accounting for spatial dependence structures and parameter uncertainty. The implementation balances statistical rigor with computational feasibility, making it suitable for the high-resolution UKCP projections.

---

## 6. Concluding Remarks

This **latent modeling** approach leverages **all** available data—both **observed** sub-daily precipitation and **simulated** precipitation from multiple UKCP runs—to estimate and **correct** biases in the simulation parameters. By **hierarchically linking** simulated parameters $\boldsymbol{\theta}_{\ell,r}^{\text{sim}}$ to the **true** physical parameters $\boldsymbol{\theta}_{\ell}^{\text{true}}$, we obtain:

- **Improved estimates** of the true climate extremes parameters,  
- **Quantified simulator biases** via $\boldsymbol{\alpha}$ and $\boldsymbol{\beta}$,  
- A **unified** framework for uncertainty propagation, enabling more **robust** sub-daily extreme precipitation forecasts throughout the 21st century.

