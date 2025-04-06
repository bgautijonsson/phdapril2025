# Midterm PhD Presentation Outline

## 1. Introduction (2–3 minutes)
- **Motivation and Context**  
  - Briefly describe the broader goal: improving projections of sub-daily extreme precipitation using advanced spatial statistical models.  
  - Highlight the importance of extreme precipitation modeling for climate projections and hydrological planning.  
  - Summarize the key theoretical underpinnings (e.g., Matérn covariance, SPDE approach, Gaussian copulas) in simple terms.

## 2. Achievements to Date (≈20 minutes)

### 2.1 Development of the Matérn-like Copula for Max-and-Smooth
- **Recap of Max-and-Smooth:**  
  - Explain the original Max-and-Smooth framework for extremes.  
  - State why integrating a Matérn-like copula can enhance modeling of spatial dependence.
- **SPDE Connection:**  
  - Show how the \((\kappa^2 - \Delta)^{\alpha/2}\) operator leads to a GMRF representation of the Matérn field.  
  - Emphasize the link between discretizing the SPDE and obtaining a precision matrix with Kronecker-sum structure.

### 2.2 Fast Computations via Kronecker Sums and Circulant Approximations
- **Kronecker Sum Basics:**  
  - Outline how 2D finite-difference (or finite-element) discretizations lead to \(\mathbf{Q}_{\rho_1} \otimes \mathbf{I} + \mathbf{I} \otimes \mathbf{Q}_{\rho_2}\).  
  - Mention the advantage of diagonalizing each 1D operator to get eigenvalues \(\lambda_{\rho_1} + \lambda_{\rho_2}\).  
- **Circulant / FFT Methods:**  
  - Illustrate how circulant embedding (or folded circulant) further speeds up determinant and quadratic-form computations.  
  - Share benchmark results comparing Cholesky vs. eigen vs. circulant approaches.

### 2.3 Handling Marginal Variances (Scaling to a Correlation Matrix)
- **Diagonal Rescaling Concept:**  
  - Summarize how the inverse of \(\mathbf{Q}\) might not be a correlation matrix by default.  
  - Show the approach to compute \(\sigma_i^2\) from the eigen-decomposition and rescale \(\mathbf{Q}\) accordingly.
- **Practical Impact:**  
  - Emphasize that this allows for proper marginal standardization without forming \(\mathbf{Q}\) or \(\mathbf{Q}^{-1}\) explicitly.  
  - Provide any timing or memory usage improvements you’ve observed.

### 2.4 Application Skeleton (Preliminary Results, if any)
- **Example or Simulation Study:**  
  - If you have any synthetic data examples or partial real-data analysis, present brief findings.  
  - Demonstrate that your method accurately captures spatial dependence in extreme precipitation scenarios.
- **Comparison to Existing Methods (if done)**  
  - Very briefly note how your approach differs or improves upon simpler geostatistical or extremes frameworks (e.g., max-stable processes, simpler copulas).

## 3. Plans for the Remainder of the PhD (≈10 minutes)

### 3.1 Bias-Correction Framework
- **Hierarchical Model for \(\Theta_{\text{true}}\) vs. \(\Theta_{\text{sim}}\):**  
  - Outline the idea that each simulation run’s parameters \(\Theta_{\text{sim}}\) is a linear function of \(\Theta_{\text{true}}\).  
  - Mention how you’ll incorporate partial pooling of multiple UKCP runs and observed data in a single Bayesian (or hierarchical) model.

### 3.2 Full Application to UKCP Data
- **Data Description:**  
  - Mention the sub-daily observed dataset (5×5 km grid) and the UKCP runs you’ll use.  
  - State anticipated challenges: large domain, missing data, nonstationarity, etc.
- **Bias-Correction Goals:**  
  - Describe your plan to recalibrate climate model outputs.  
  - Highlight expected outcomes: improved sub-daily extremes projection, better uncertainty quantification.

### 3.3 Software Package and Dissemination
- **R Package Development:**  
  - Confirm your intention to package the Max-and-Smooth + Matérn-like copula approach for broader usage.  
  - Outline high-level features (e.g., user interface, parallel or C++ integration, vignettes).
- **Manuscript Pipeline:**  
  - Recap your three-paper structure: (1) Theoretical dev + new copula, (2) Real-data application, (3) Software implementation.

## 4. Conclusion and Discussion
- **Key Takeaways:**  
  - Summarize how the SPDE + GMRF + copula synergy addresses spatial extremes more efficiently and flexibly.  
  - Restate the unique contributions (fast computations, bias-correction via hierarchical modeling, open-source software).
- **Open Questions / Next Steps:**  
  - Mention any methodological or computational uncertainties that you intend to tackle (e.g., higher-dimensional grids, nonstationarity, GPU parallelization).
- **Q&A Invitation:**  
  - Invite feedback or questions from committee members and the audience.

---

**Thank You**  
> *Slide: “Thanks! Questions?”*

