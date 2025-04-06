#' @export
dmatern_copula <- function(X, dim, rho, nu) {
  if (is.vector(X) | (is.atomic(X) & NCOL(X) == 1)) {
    X <- matrix(X, ncol = 1)
  }

  if (length(rho) == 1) {
    return(dmatern_copula_eigen(X, dim, dim, rho, rho, nu))
  }

  dmatern_copula_eigen(X, dim[1], dim[2], rho[1], rho[2], nu)
}

#' @export
dmatern <- function(X, dim, rho, nu) {
  if (is.vector(X) | (is.atomic(X) & NCOL(X) == 1)) {
    X <- matrix(X, ncol = 1)
  }

  if (length(rho) == 1) {
    return(dmatern_eigen(X, dim, dim, rho, rho, nu))
  }

  dmatern_eigen(X, dim[1], dim[2], rho[1], rho[2], nu)
}

#' @export
rmatern_copula <- function(n, dim, rho, nu) {
  if (length(rho) == 1) {
    return(rmatern_copula_eigen(n, dim, dim, rho, rho, nu))
  }
  rmatern_copula_eigen(n, dim[1], dim[2], rho[1], rho[2], nu)
}

#' @export
rmatern <- function(n, dim, rho, nu) {
  if (length(rho) == 1) {
    return(rmatern_eigen(n, dim, dim, rho, rho, nu))
  }
  rmatern_eigen(n, dim[1], dim[2], rho[1], rho[2], nu)
}

#' @export
make_standardized_matern <- function(dim, rho, nu) {
  if (length(rho) == 1) {
    return(make_standardized_matern_eigen(dim, dim, rho, rho, nu))
  }
  make_standardized_matern_eigen(dim[1], dim[2], rho[1], rho[2], nu)
}