#!/usr/bin/env Rscript
# ==============================================================================
# time_roc_analysis.R
# ==============================================================================
# Time-dependent ROC analysis for disease-free survival (DFS) prediction
# in small-cell lung cancer cohorts.
#
# Produces two figures:
#   1. Combined 1/3/5-year ROC per cohort  (2x2 layout)
#   2. Individual ROC with bootstrap CI     (3x4 layout)
#
# Usage:
#   source("time_roc_analysis.R")
#   results <- run_roc_analysis(data_dir, save_dir)
# ==============================================================================

# ---- Dependencies ------------------------------------------------------------
suppressPackageStartupMessages({
  library(dplyr)
  library(survival)
  library(timeROC)
  library(ggplot2)
  library(ggpubr)
})

# ==============================================================================
# 1. Configuration
# ==============================================================================

#' Default analysis settings (modify here or override via function args)
default_config <- function() {
  list(
    # Column names in input CSVs
    time_col   = "DFS",
    event_col  = "DFSState",
    marker_col = "PreHazard",

    # Time points in months
    time_points   = c(12, 36, 60),
    time_labels   = c("1-Year", "3-Year", "5-Year"),

    # NPG palette (Nature Publishing Group)
    palette = c("#E64B35FF", "#4DBBD5FF", "#00A087FF"),

    # Bootstrap settings (individual ROC with CI bands)
    n_boot     = 50,
    boot_seed  = 123,
    grid_len   = 500,

    # Output dimensions (inches)
    combined_size   = c(width = 5, height = 5),
    individual_size = c(width = 7.8, height = 7.8),
    dpi = 300
  )
}


# ==============================================================================
# 2. Data Helpers
# ==============================================================================

#' Load all cohort CSVs from a directory.
#'
#' @param data_dir  Path containing the four CSV files.
#' @return Named list of data.frames.
load_cohorts <- function(data_dir) {
  file_map <- c(
    "CHCAMS Train"    = "CHCAMS_Train.csv",
    "CHCAMS Val"      = "CHCAMS_Val.csv",
    "HMUCH External"  = "HMUCH_Val.csv",
    "TMUGH External"  = "TMUGH_Val.csv"
  )
  cohorts <- lapply(file_map, function(f) {
    path <- file.path(data_dir, f)
    if (!file.exists(path)) stop("File not found: ", path)
    read.csv(path, encoding = "utf-8")
  })
  names(cohorts) <- names(file_map)
  cohorts
}


#' Drop rows with NA in required columns.
#'
#' @param df   A data.frame.
#' @param cfg  Config list (needs time_col, event_col, marker_col).
#' @return Filtered data.frame.
clean_roc_data <- function(df, cfg) {
  cols <- c(cfg$time_col, cfg$event_col, cfg$marker_col)
  df %>% filter(if_all(all_of(cols), ~ !is.na(.)))
}


# ==============================================================================
# 3. ROC Computation Helpers
# ==============================================================================

#' Compute timeROC and extract AUC with 95% CI.
#'
#' @param df          Cleaned data.frame.
#' @param time_points Numeric vector of time horizons (months).
#' @param cfg         Config list.
#' @return A timeROC object (with $auc_table attached for convenience).
compute_time_roc <- function(df, time_points, cfg) {
  roc <- timeROC(
    T        = df[[cfg$time_col]],
    delta    = df[[cfg$event_col]],
    marker   = df[[cfg$marker_col]],
    cause    = 1,
    weighting = "marginal",
    times    = time_points,
    iid      = TRUE
  )
  # Attach a tidy AUC summary table
  ci <- confint(roc, level = 0.95)$CI_AUC / 100
  roc$auc_table <- data.frame(
    time  = time_points,
    auc   = roc$AUC[-1],
    lower = ci[, 1],
    upper = ci[, 2]
  )
  roc
}


#' Interpolate an ROC curve onto a regular FPR grid, anchored at (0,0)/(1,1).
#'
#' @param fp     Raw false-positive rates from timeROC.
#' @param tp     Raw true-positive rates from timeROC.
#' @param x_grid Regular grid of FPR values.
#' @return Interpolated TPR vector aligned to x_grid.
interpolate_roc <- function(fp, tp, x_grid) {
  valid <- is.finite(fp) & is.finite(tp)
  fp <- c(0, fp[valid], 1)
  tp <- c(0, tp[valid], 1)
  ord <- order(fp, tp)
  y <- approx(fp[ord], tp[ord], xout = x_grid, rule = 2, ties = max)$y
  y[1] <- 0
  y
}


# ==============================================================================
# 4. Plot: Combined Multi-Year ROC (2x2)
# ==============================================================================

#' Plot 1/3/5-year ROC curves on a single panel (one cohort).
#'
#' @param roc_obj   timeROC result from compute_time_roc().
#' @param title     Panel title string.
#' @param cfg       Config list.
#' @return A ggplot object.
plot_combined_roc <- function(roc_obj, title, cfg) {

  n_tp <- length(cfg$time_points)
  at <- roc_obj$auc_table

  # Build legend labels:  "1-Year AUC: 0.850 (0.810-0.890)"
  legend_labels <- sprintf(
    "%s AUC: %.3f (%.3f\u2013%.3f)",
    cfg$time_labels, at$auc, at$lower, at$upper
  )

  # Stack FP/TP for all time points (columns 2..n in roc_obj$FP / $TP)
  plot_df <- do.call(rbind, lapply(seq_len(n_tp), function(i) {
    data.frame(
      FP    = roc_obj$FP[, i + 1],
      TP    = roc_obj$TP[, i + 1],
      Group = cfg$time_labels[i]
    )
  }))
  plot_df$Group <- factor(plot_df$Group,
                          levels = cfg$time_labels,
                          labels = legend_labels)

  ggplot(plot_df, aes(FP, TP, color = Group)) +
    geom_line(linewidth = 1.2) +
    geom_abline(slope = 1, intercept = 0,
                linetype = "dashed", color = "grey60", linewidth = 0.6) +
    scale_color_manual(values = cfg$palette) +
    scale_x_continuous(limits = c(0, 1), expand = c(0.01, 0.01)) +
    scale_y_continuous(limits = c(0, 1), expand = c(0.01, 0.01)) +
    coord_fixed() +
    labs(title = title, x = "False Positive Rate",
         y = "True Positive Rate", color = NULL) +
    theme_bw() +
    theme(
      plot.title       = element_text(hjust = 0.5, size = 10),
      axis.title       = element_text(size = 10),
      axis.text        = element_text(size = 8, color = "black"),
      legend.position  = c(0.60, 0.18),
      legend.background = element_blank(),
      legend.key        = element_blank(),
      legend.text       = element_text(size = 8),
      panel.grid.major  = element_line(color = "grey92"),
      panel.grid.minor  = element_blank()
    )
}


# ==============================================================================
# 5. Plot: Individual ROC with Bootstrap CI Band (3x4)
# ==============================================================================

#' Plot a single time-point ROC with bootstrap confidence band.
#'
#' @param df           Cleaned data.frame.
#' @param time_point   Single numeric time horizon (months).
#' @param title        Panel title.
#' @param color        Hex colour string.
#' @param cfg          Config list.
#' @param show_xlab    Whether to display x-axis title.
#' @param show_ylab    Whether to display y-axis title.
#' @return A ggplot object.
plot_individual_roc <- function(df, time_point, title, color, cfg,
                                show_xlab = TRUE, show_ylab = TRUE) {

  x_grid <- seq(0, 1, length.out = cfg$grid_len)

  # -- Main curve & AUC text --
  roc_main <- compute_time_roc(df, time_point, cfg)
  at <- roc_main$auc_table
  auc_text <- sprintf("AUC: %.3f\n(%.3f\u2013%.3f)", at$auc, at$lower, at$upper)
  main_tpr <- interpolate_roc(roc_main$FP[, 2], roc_main$TP[, 2], x_grid)

  # -- Bootstrap CI band --
  set.seed(cfg$boot_seed)
  boot_mat <- matrix(NA_real_, nrow = cfg$n_boot, ncol = length(x_grid))

  for (b in seq_len(cfg$n_boot)) {
    idx <- sample(nrow(df), replace = TRUE)
    tryCatch({
      r <- timeROC(
        T = df[[cfg$time_col]][idx],
        delta = df[[cfg$event_col]][idx],
        marker = df[[cfg$marker_col]][idx],
        cause = 1, weighting = "marginal",
        times = time_point, iid = FALSE
      )
      boot_mat[b, ] <- interpolate_roc(r$FP[, 2], r$TP[, 2], x_grid)
    }, error = function(e) NULL)
  }

  ci_lo <- apply(boot_mat, 2, quantile, probs = 0.025, na.rm = TRUE)
  ci_hi <- apply(boot_mat, 2, quantile, probs = 0.975, na.rm = TRUE)

  plot_df <- data.frame(FP = x_grid, TP = main_tpr,
                        CI_Lo = ci_lo, CI_Hi = ci_hi)

  # -- Assemble plot --
  p <- ggplot(plot_df, aes(x = FP)) +
    geom_ribbon(aes(ymin = CI_Lo, ymax = CI_Hi), fill = color, alpha = 0.2) +
    geom_line(aes(y = TP), color = color, linewidth = 1.0) +
    geom_abline(slope = 1, intercept = 0,
                linetype = "dashed", color = "grey60", linewidth = 0.5) +
    annotate("text", x = 0.70, y = 0.15, label = auc_text,
             size = 3.0, hjust = 0.5, lineheight = 0.9) +
    scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
    scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
    coord_fixed() +
    labs(title = title, x = "1 - Specificity", y = "Sensitivity") +
    theme_bw() +
    theme(
      plot.title  = element_text(hjust = 0.5, size = 10, face = "bold"),
      axis.title  = element_text(size = 8),
      axis.text   = element_text(size = 8, color = "black"),
      panel.grid  = element_blank()
    )

  if (!show_xlab) p <- p + theme(axis.title.x = element_blank())
  if (!show_ylab) p <- p + theme(axis.title.y = element_blank())
  p
}


# ==============================================================================
# 6. Main Entry Point
# ==============================================================================

#' Run the complete ROC analysis pipeline.
#'
#' @param data_dir  Directory containing the four cohort CSVs.
#' @param save_dir  Directory for saving output figures.
#' @param cfg       Optional config list (default: default_config()).
#' @return Invisible list of ggplot objects.
run_roc_analysis <- function(data_dir, save_dir, cfg = default_config()) {

  dir.create(save_dir, showWarnings = FALSE, recursive = TRUE)
  cohorts <- load_cohorts(data_dir)
  panel_prefix <- c("A", "B", "C", "D")

  # ---- Figure 1: Combined 1/3/5-year ROC (2x2) ----------------------------
  message(">> Generating combined ROC plots ...")
  combined_plots <- mapply(function(df, name, prefix) {
    df_clean <- clean_roc_data(df, cfg)
    roc_obj  <- compute_time_roc(df_clean, cfg$time_points, cfg)
    plot_combined_roc(roc_obj, paste0(prefix, ". ", name), cfg)
  }, cohorts, names(cohorts), panel_prefix, SIMPLIFY = FALSE)

  fig_combined <- ggarrange(plotlist = combined_plots, ncol = 2, nrow = 2)
  ggsave(file.path(save_dir, "Figure_ROC_Combined_2x2.pdf"),
         plot = fig_combined,
         width = cfg$combined_size["width"],
         height = cfg$combined_size["height"],
         dpi = cfg$dpi)
  message("   Saved Figure_ROC_Combined_2x2.pdf")

  # ---- Figure 2: Individual ROC with CI bands (3x4) -----------------------
  message(">> Generating individual ROC plots (bootstrap) ...")
  n_years   <- length(cfg$time_points)
  n_cohorts <- length(cohorts)
  ind_plots <- vector("list", n_years * n_cohorts)
  idx <- 1

  for (i in seq_len(n_years)) {
    for (j in seq_len(n_cohorts)) {
      cohort_name <- names(cohorts)[j]
      title_str   <- paste0(cohort_name, " (", cfg$time_labels[i], ")")
      message(sprintf("   %s", title_str))

      df_clean <- clean_roc_data(cohorts[[j]], cfg)
      ind_plots[[idx]] <- plot_individual_roc(
        df        = df_clean,
        time_point = cfg$time_points[i],
        title     = title_str,
        color     = cfg$palette[i],
        cfg       = cfg,
        show_xlab = (i == n_years),
        show_ylab = (j == 1)
      )
      idx <- idx + 1
    }
  }

  fig_individual <- ggarrange(plotlist = ind_plots,
                              ncol = n_cohorts, nrow = n_years,
                              labels = "AUTO")
  ggsave(file.path(save_dir, "Figure_ROC_Individual_3x4.pdf"),
         plot = fig_individual,
         width = cfg$individual_size["width"],
         height = cfg$individual_size["height"],
         dpi = cfg$dpi)
  ggsave(file.path(save_dir, "Figure_ROC_Individual_3x4.png"),
         plot = fig_individual,
         width = cfg$individual_size["width"],
         height = cfg$individual_size["height"],
         dpi = cfg$dpi)
  message("   Saved Figure_ROC_Individual_3x4.pdf/.png")

  message(">> All done.")
  invisible(list(combined = fig_combined, individual = fig_individual))
}


# ==============================================================================
# 7. CLI Entry (runs only when script is executed directly)
# ==============================================================================
if (!interactive() || identical(environment(), globalenv())) {
  # ---- Modify these two paths to match your machine ----
  DATA_DIR <- "x"
  SAVE_DIR <- "xx"

  # Optional: override any default thresholds
  cfg <- default_config()
  # cfg$n_boot <- 200           # increase for publication
  # cfg$palette <- c(...)       # swap colours

  run_roc_analysis(DATA_DIR, SAVE_DIR, cfg)
}
