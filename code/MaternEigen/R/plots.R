#' @export
plot_matern <- function(X, dim1, dim2) {
  dplyr::tibble(
    X = as.numeric(X)
  ) |> 
    dplyr::mutate(
      id = dplyr::row_number(),
      lat = rep(seq_len(dim1), each = dim2),
      lon = rep(seq_len(dim2), times = dim1)
    ) |> 
    ggplot2::ggplot(ggplot2::aes(lat, lon, fill = X)) +
    ggplot2::geom_raster() +
    ggplot2::scale_fill_viridis_c() +
    ggplot2::coord_fixed(expand = FALSE) +
    ggplot2::labs(
      x = NULL,
      y = NULL,
      fill = NULL
    ) +
    ggplot2::theme(
      legend.position = "none"
    )
}