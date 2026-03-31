#!/usr/bin/env Rscript
# ==============================================================================
# metastasis_forest_plot.R
# ==============================================================================
# Pooled logistic-regression analysis of AI risk score vs. site-specific
# postoperative metastasis in small-cell lung cancer.
#
# For each metastasis site the script computes:
#   - Odds Ratio (OR) with 95 % CI  (logistic regression)
#   - AUC with 95 % CI              (DeLong method via pROC)
#
# Output: a composite forest-plot + summary-table figure (PDF).
#
# Usage:
#   source("metastasis_forest_plot.R")
#   run_metastasis_analysis(data_dir, save_dir)
# ==============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(pROC)
  library(broom)
  library(gridExtra)
  library(grid)
})

# ==============================================================================
# 1. Configuration
# ==============================================================================

default_config <- function() {
  list(
    # CSV file-name mapping (order: train, val, external1, external2)
    cohort_files = c(
      "CHCAMS_Train" = "CHCAMS_Train_M.csv",
      "CHCAMS_Val"   = "CHCAMS_Val_M.csv",
      "HMUCH_Val"    = "HMUCH_Val_M.csv",
      "TMUGH_Val"    = "TMUGH_Val_M.csv"
    ),

    # Predictor column
    marker_col = "PreHazard",

    # Outcome columns -> display labels (ordered top-to-bottom in plot)
    outcomes = c(
      "Postoperative.Lymphatic.Metastasis" = "Lymphatic Metastasis",
      "Postoperative.Brain.Metastasis"     = "Brain Metastasis",
      "Postoperative.Liver.Metastasis"     = "Liver Metastasis",
      "Postoperative.Bone.Metastasis"      = "Bone Metastasis"
    ),

    # NPG-inspired per-site colours
    palette = c(
      "Lymphatic Metastasis" = "#E64B35",
      "Brain Metastasis"     = "#4DBBD5",
      "Liver Metastasis"     = "#00A087",
      "Bone Metastasis"      = "#3C5488"
    ),

    # Minimum positive-event count to run analysis
    min_events = 5,

    # Output figure dimensions (inches)
    fig_width  = 5,
    fig_height = 2
  )
}


# ==============================================================================
# 2. Data Loading
# ==============================================================================

#' Load cohort CSVs, keep only marker + outcome columns, and pool all rows.
#'
#' @return A single data.frame (pooled cohort).
load_pooled_data <- function(data_dir, cfg) {
  needed <- c(cfg$marker_col, names(cfg$outcomes))

  dfs <- lapply(cfg$cohort_files, function(f) {
    read.csv(file.path(data_dir, f), encoding = "utf-8") %>%
      dplyr::select(dplyr::any_of(needed)) %>%
      dplyr::mutate(across(dplyr::any_of(names(cfg$outcomes)), as.numeric))
  })

  pooled <- bind_rows(dfs)
  message(sprintf("  Pooled sample size: %d", nrow(pooled)))
  pooled
}


# ==============================================================================
# 3. Core Analysis (OR + AUC per outcome)
# ==============================================================================

#' Run logistic regression + ROC for one binary outcome.
#'
#' @return A one-row data.frame, or NULL if too few events.
analyse_one_outcome <- function(pooled, outcome_col, outcome_label, cfg) {
  df <- pooled %>%
    dplyr::select(all_of(c(cfg$marker_col, outcome_col))) %>%
    drop_na()

  if (sum(df[[outcome_col]] == 1, na.rm = TRUE) < cfg$min_events) {
    warning(outcome_col, ": fewer than ", cfg$min_events, " events — skipped.")
    return(NULL)
  }

  # Logistic regression -> OR
  mdl <- glm(as.formula(paste(outcome_col, "~", cfg$marker_col)),
              data = df, family = binomial())
  or_res <- tidy(mdl, exponentiate = TRUE, conf.int = TRUE) %>%
    filter(term == cfg$marker_col)

  # ROC -> AUC (DeLong CI)
  roc_obj <- roc(df[[outcome_col]], df[[cfg$marker_col]], quiet = TRUE)
  ci_auc  <- ci(roc_obj)

  data.frame(
    Outcome_Key   = outcome_col,
    Outcome_Label = outcome_label,
    OR      = or_res$estimate,
    OR_Low  = or_res$conf.low,
    OR_High = or_res$conf.high,
    P_Value = or_res$p.value,
    AUC     = as.numeric(auc(roc_obj)),
    AUC_Low = ci_auc[1],
    AUC_High = ci_auc[3],
    row.names = NULL
  )
}


#' Iterate over all outcomes and return a combined results table.
analyse_all_outcomes <- function(pooled, cfg) {
  rows <- mapply(
    analyse_one_outcome,
    outcome_col   = names(cfg$outcomes),
    outcome_label = unname(cfg$outcomes),
    MoreArgs = list(pooled = pooled, cfg = cfg),
    SIMPLIFY = FALSE
  )
  do.call(rbind, Filter(Negate(is.null), rows))
}


# ==============================================================================
# 4. Formatting Helpers
# ==============================================================================

#' Add display-ready text columns and set factor order for plotting.
format_results <- function(results, cfg) {
  label_order <- rev(unname(cfg$outcomes))  # top-to-bottom in plot

  results %>% dplyr::mutate(
    P_text  = ifelse(P_Value < 0.001, "< 0.001", sprintf("%.3f", P_Value)),
    OR_text = sprintf("%.2f (%.2f\u2013%.2f)", OR, OR_Low, OR_High),
    AUC_text = sprintf("%.3f (%.3f\u2013%.3f)", AUC, AUC_Low, AUC_High),
    Outcome_Label = factor(Outcome_Label, levels = label_order)
  )
}


# ==============================================================================
# 5. Plotting
# ==============================================================================

#' Left panel: forest plot (OR with error bars on log scale).
build_forest_panel <- function(df, cfg) {
  ggplot(df, aes(x = OR, y = Outcome_Label, color = Outcome_Label)) +
    geom_vline(xintercept = 1, linetype = "dashed", color = "gray60") +
    geom_errorbarh(aes(xmin = OR_Low, xmax = OR_High),
                   height = 0.2, linewidth = 0.8) +
    geom_point(size = 4, shape = 18) +
    scale_color_manual(values = cfg$palette) +
    scale_x_log10(breaks = c(0.5, 1, 2, 5, 10)) +
    labs(x = "Odds Ratio (log scale)", y = NULL) +
    theme_bw() +
    theme(
      legend.position  = "none",
      axis.text.y      = element_text(face = "bold", size = 8, color = "black"),
      axis.text.x      = element_text(size = 8),
      panel.grid.minor = element_blank(),
      panel.grid.major.y = element_blank(),
      plot.margin = margin(r = 0)
    )
}


#' Right panel: summary table (AUC, OR, P-value as text columns).
build_table_panel <- function(df) {
  ggplot(df, aes(y = Outcome_Label)) +
    geom_text(aes(x = 1,   label = AUC_text), size = 3.5) +
    geom_text(aes(x = 2.5, label = OR_text),  size = 3.5) +
    geom_text(aes(x = 4,   label = P_text,
                  fontface = ifelse(P_Value < 0.05, "bold", "plain")),
              size = 3.5) +
    scale_x_continuous(
      limits = c(0.5, 4.5),
      breaks = c(1, 2.5, 4),
      labels = c("AUC (95% CI)", "OR (95% CI)", "P Value")
    ) +
    theme_void() +
    theme(
      axis.text.x = element_text(face = "bold", size = 11, color = "black",
                                  margin = margin(b = 10)),
      plot.margin = margin(l = 0)
    )
}


#' Combine forest + table panels with a shared title.
compose_figure <- function(df, cfg) {
  grid.arrange(
    build_forest_panel(df, cfg),
    build_table_panel(df),
    ncol   = 2,
    widths = c(1.5, 1.5),
    top    = textGrob(
      "Pooled Analysis: Association of Risk Score with Metastasis Sites",
      gp = gpar(fontsize = 15, fontface = "bold")
    )
  )
}


# ==============================================================================
# 6. Main Entry Point
# ==============================================================================

#' Run the full metastasis-association pipeline.
#'
#' @param data_dir  Directory containing the four *_M.csv files.
#' @param save_dir  Directory for the output PDF.
#' @param cfg       Config list (default: default_config()).
#' @return Invisible results data.frame.
run_metastasis_analysis <- function(data_dir, save_dir, cfg = default_config()) {

  dir.create(save_dir, showWarnings = FALSE, recursive = TRUE)

  message(">> Loading & pooling cohort data ...")
  pooled <- load_pooled_data(data_dir, cfg)

  message(">> Running logistic regression + ROC per outcome ...")
  results <- analyse_all_outcomes(pooled, cfg)
  print(results %>% dplyr::select(-Outcome_Key))

  message(">> Building forest plot ...")
  plot_df <- format_results(results, cfg)
  fig     <- compose_figure(plot_df, cfg)

  out_path <- file.path(save_dir, "Pooled_Metastasis_ForestPlot.pdf")
  ggsave(out_path, fig,
         width = cfg$fig_width, height = cfg$fig_height)
  message("  Saved: ", out_path)

  message(">> Done.")
  invisible(results)
}


# ==============================================================================
# 7. CLI Entry
# ==============================================================================
if (!interactive() || identical(environment(), globalenv())) {
  DATA_DIR <- "xx"
  SAVE_DIR <- "xxx"

  run_metastasis_analysis(DATA_DIR, SAVE_DIR)
}
