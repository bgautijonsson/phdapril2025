#' Run the Max step of the Max-and-Smooth algorithm
#'
#' @param Y A matrix of precipitation data where rows are observations and columns are locations
#' @param n_x Number of x-axis grid points
#' @param n_y Number of y-axis grid points
#' @param nu Smoothness parameter for the Matern covariance (default = 2)
#'
#' @return A list containing:
#'   \item{parameters}{Estimated GEV parameters}
#'   \item{Hessian}{Hessian matrix of the fit}
#'   \item{L}{Cholesky factor of precision matrix}
#'   \item{stations}{Data frame with station information and parameter estimates}
#'
#' @importFrom tidyr pivot_wider
#' @importFrom dplyr mutate tibble inner_join
#' @export
ms_max <- function(Y, n_x, n_y, nu = 2) {
  # Step 1: Perform MLE for each location
  res_iid <- max_iid(Y, "gev")

  # Step 2: Fit the Matern copula
  rho <- fit_copula(Y, n_x, n_y, nu)

  # Step 3: Prepare precision matrix components
  precision_components <- prepare_precision(
    rho,
    n_x,
    n_y,
    nu
  )



  # Step 4: Fit GEV distribution using results from step 1 and 3
  res_copula <- max_copula(
    data = Y,
    index = precision_components$index,
    n_values = precision_components$n_values,
    values = precision_components$values,
    log_det = precision_components$log_det,
    init = res_iid$eta_hat
  )


  # Return results
  list(
    parameters_copula = res_copula$parameters,
    parameters_iid = res_iid$eta_hat,
    Hessian = res_copula$Hessian,
    L = res_copula$L,
    rho = rho
  )
}
