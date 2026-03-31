#!/usr/bin/env Rscript
# ==============================================================================
# timedep_auc_comparison.R
# ==============================================================================
# C-index evaluation and time-dependent AUC bar chart for three Cox models
# (AI-only, Clinical-only, Combined) across multiple SCLC cohorts.
#
# Output:
#   - Console table of C-index + pairwise p-values
#   - 2x2 grouped bar chart of 1/3/5-year AUC per cohort (NEJM palette)
#
# Usage:
#   source("timedep_auc_comparison.R")
#   run_timedep_auc_analysis(data_dir, save_dir)
# ==============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(ggpubr)
  library(ggsci)
  library(survival)
  library(survminer)
  library(timeROC)
  library(rms)
  library(Hmisc)
  library(compareC)
})

# ==============================================================================
# 1. Configuration
# ==============================================================================

default_config <- function() {
  list(
    cohort_files = c(
      "CHCAMS_Train" = "CHCAMS_Train.csv",
      "CHCAMS_Val"   = "CHCAMS_Val.csv",
      "HMUCH_Val"    = "HMUCH_Val.csv",
      "TMUGH_Val"    = "TMUGH_Val.csv"
    ),

    time_col   = "DFS",
    event_col  = "DFSState",
    marker_col = "PreHazard",

    # Factor-coding rules
    factor_spec = list(
      Gender         = list(levels = c(1, 2),    labels = c("Male", "Female")),
      SmokingHistory = list(levels = c(0, 1),    labels = c("No", "Yes")),
      AJCCStage      = list(levels = c(1, 2, 3), labels = c("Stage I", "Stage II", "Stage III"))
    ),

    # Time-dependent AUC horizons (months)
    time_points = c(12, 36, 60),
    time_labels = c("1-Year", "3-Year", "5-Year"),

    # Three model names (display order in legend)
    model_names = c("Stage Model", "Risk Model", "Combined Model"),

    # Output
    fig_width  = 9,
    fig_height = 7
  )
}


# ==============================================================================
# 2. Data Helpers (shared with cindex_nomogram_analysis.R)
# ==============================================================================

#' Load cohort CSVs, drop rows with missing AJCCStage.
load_cohorts <- function(data_dir, cfg) {
  lapply(cfg$cohort_files, function(f) {
    read.csv(file.path(data_dir, f), encoding = "utf-8") %>%
      dplyr::filter(!is.na(AJCCStage))
  })
}

#' Apply factor coding and numeric coercion.
clean_cohort <- function(df, cfg) {
  for (col in names(cfg$factor_spec)) {
    spec <- cfg$factor_spec[[col]]
    df[[col]] <- factor(df[[col]], levels = spec$levels, labels = spec$labels)
  }
  df %>% dplyr::mutate(
    Age       = as.numeric(Age),
    PreHazard = as.numeric(PreHazard),
    DFS       = as.numeric(DFS),
    DFSState  = as.numeric(DFSState)
  )
}

#' Determine optimal cut-off on training set and assign risk groups.
assign_risk_groups <- function(cohorts, cfg) {
  cp <- surv_cutpoint(cohorts[[1]],
                      time = cfg$time_col, event = cfg$event_col,
                      variables = cfg$marker_col, progressbar = FALSE)
  cutoff <- cp$cutpoint$cutpoint
  message(sprintf("  Risk cut-off: %.4f", cutoff))

  lapply(cohorts, function(df) {
    df$HR        <- ifelse(df[[cfg$marker_col]] > cutoff, "High_Risk", "Low_Risk")
    df$Pre_Label <- factor(df$HR, levels = c("High_Risk", "Low_Risk"))
    df
  })
}


# ==============================================================================
# 3. Model Fitting
# ==============================================================================

#' Fit three Cox models on training data; return named list.
fit_cox_models <- function(train_df) {
  dd <- datadist(train_df)
  options(datadist = "dd")

  list(
    "Risk Model" = cph(Surv(DFS, DFSState) ~ PreHazard,
                       data = train_df, x = TRUE, y = TRUE, surv = TRUE),

    "Stage Model" = cph(Surv(DFS, DFSState) ~ AJCCStage + Gender + SmokingHistory + Age,
                        data = train_df, x = TRUE, y = TRUE, surv = TRUE),

    "Combined Model" = cph(Surv(DFS, DFSState) ~ PreHazard + AJCCStage + Gender + SmokingHistory + Age,
                           data = train_df, x = TRUE, y = TRUE, surv = TRUE)
  )
}


# ==============================================================================
# 4. C-index Evaluation
# ==============================================================================

calc_cindex <- function(lp, time, status) {
  x  <- Hmisc::rcorr.cens(-lp, Surv(time, status))
  ci <- x["C Index"]; se <- x["S.D."] / 2
  c(C = ci, Low = ci - 1.96 * se, High = ci + 1.96 * se)
}

evaluate_cindex <- function(df, cohort_name, models) {
  lps <- lapply(models, function(m) predict(m, newdata = df, type = "lp"))
  res <- lapply(lps, function(l) calc_cindex(l, df$DFS, df$DFSState))

  p_rv <- compareC(df$DFS, df$DFSState, lps[["Risk Model"]],  lps[["Stage Model"]])$pval
  p_cv <- compareC(df$DFS, df$DFSState, lps[["Combined Model"]], lps[["Stage Model"]])$pval

  data.frame(
    Cohort = cohort_name,
    Risk_C = res[["Risk Model"]]["C"],       Risk_Low = res[["Risk Model"]]["Low"],       Risk_High = res[["Risk Model"]]["High"],
    Stage_C = res[["Stage Model"]]["C"],      Stage_Low = res[["Stage Model"]]["Low"],      Stage_High = res[["Stage Model"]]["High"],
    Comb_C = res[["Combined Model"]]["C"],    Comb_Low = res[["Combined Model"]]["Low"],    Comb_High = res[["Combined Model"]]["High"],
    P_Risk_vs_Stage = p_rv, P_Comb_vs_Stage = p_cv,
    row.names = NULL
  )
}


# ==============================================================================
# 5. Time-dependent AUC Computation
# ==============================================================================

#' Compute time-dependent AUC (+ 95 % CI) for one model on one cohort.
compute_timedep_auc <- function(df, cohort_name, model, model_name, cfg) {
  lp <- predict(model, newdata = df, type = "lp")

  roc_res <- timeROC(
    T = df[[cfg$time_col]], delta = df[[cfg$event_col]],
    marker = lp, cause = 1,
    weighting = "marginal", times = cfg$time_points, iid = TRUE
  )

  auc_vals <- roc_res$AUC[match(cfg$time_points, roc_res$times)]
  se_vals  <- roc_res$inference$vect_sd_1[match(cfg$time_points, roc_res$times)]

  data.frame(
    Cohort    = cohort_name,
    Model     = model_name,
    TimePoint = cfg$time_labels,
    AUC       = auc_vals,
    Lower     = pmax(auc_vals - 1.96 * se_vals, 0),
    Upper     = pmin(auc_vals + 1.96 * se_vals, 1),
    row.names = NULL
  )
}


#' Iterate over all cohorts x models; return a single long data.frame.
compute_all_timedep_auc <- function(cohorts, models, cfg) {
  rows <- list()
  for (cn in names(cohorts)) {
    for (mn in names(models)) {
      rows[[paste(cn, mn)]] <- compute_timedep_auc(
        cohorts[[cn]], cn, models[[mn]], mn, cfg
      )
    }
  }
  out <- do.call(rbind, rows)
  out$Cohort    <- factor(out$Cohort,    levels = names(cohorts))
  out$Model     <- factor(out$Model,     levels = cfg$model_names)
  out$TimePoint <- factor(out$TimePoint, levels = cfg$time_labels)
  out
}


# ==============================================================================
# 6. Plotting (NEJM-style grouped bar chart, 2x2)
# ==============================================================================

#' Draw one cohort panel.
plot_auc_panel <- function(cohort_name, auc_df) {
  df <- auc_df %>% dplyr::filter(Cohort == cohort_name)

  ggplot(df, aes(x = TimePoint, y = AUC, fill = Model)) +
    geom_col(position = position_dodge(0.8), width = 0.7,
             color = "black", linewidth = 0.4, alpha = 0.9) +
    geom_errorbar(aes(ymin = Lower, ymax = Upper),
                  position = position_dodge(0.8), width = 0.2,
                  linewidth = 0.5) +
    geom_text(aes(y = Upper + 0.015, label = sprintf("%.3f", AUC)),
              position = position_dodge(0.8),
              vjust = 0, size = 3.2, color = "grey10") +
    coord_cartesian(ylim = c(0.4, 1.05)) +
    scale_y_continuous(breaks = seq(0.4, 1.0, 0.2),
                       expand = expansion(mult = c(0, 0.05))) +
    scale_fill_nejm() +
    labs(title = cohort_name, y = "Time-dependent AUC", x = NULL) +
    theme_classic(base_size = 8) +
    theme(
      plot.title   = element_text(face = "bold", hjust = 0.5, size = 8,
                                   margin = margin(b = 15)),
      axis.text.x  = element_text(color = "black", face = "bold", size = 8,
                                   margin = margin(t = 5)),
      axis.text.y  = element_text(color = "black", size = 8),
      axis.title.y = element_text(face = "bold", size = 8,
                                   margin = margin(r = 10)),
      axis.line    = element_line(color = "black", linewidth = 0.7),
      axis.ticks   = element_line(color = "black", linewidth = 0.7),
      axis.ticks.length = unit(0.2, "cm"),
      legend.position = "none",
      plot.margin  = margin(15, 15, 15, 15)
    )
}


#' Assemble 2x2 layout with shared legend.
compose_auc_figure <- function(auc_df) {
  panels <- lapply(levels(auc_df$Cohort), plot_auc_panel, auc_df = auc_df)

  ggarrange(
    plotlist     = panels,
    ncol = 2, nrow = 2,
    labels       = c("A", "B", "C", "D"),
    font.label   = list(size = 8, face = "bold"),
    common.legend = TRUE,
    legend        = "bottom"
  )
}


# ==============================================================================
# 7. Main Entry Point
# ==============================================================================

#' Run the full pipeline: load -> clean -> fit -> C-index -> AUC chart.
#'
#' @param data_dir  Directory with cohort CSVs.
#' @param save_dir  Directory for output figures.
#' @param cfg       Config list (default: default_config()).
#' @return Invisible list of C-index table and AUC data.
run_timedep_auc_analysis <- function(data_dir, save_dir, cfg = default_config()) {

  dir.create(save_dir, showWarnings = FALSE, recursive = TRUE)

  # ---- Data ----
  message(">> Loading & cleaning cohorts ...")
  cohorts <- load_cohorts(data_dir, cfg) |> lapply(clean_cohort, cfg = cfg)
  cohorts <- assign_risk_groups(cohorts, cfg)

  # ---- Models ----
  message(">> Fitting Cox models on training cohort ...")
  models <- fit_cox_models(cohorts[[1]])

  # ---- C-index ----
  message(">> Evaluating C-index ...")
  cindex_df <- do.call(rbind, mapply(
    evaluate_cindex, cohorts, names(cohorts),
    MoreArgs = list(models = models), SIMPLIFY = FALSE
  ))
  print(cindex_df)

  # ---- Time-dependent AUC ----
  message(">> Computing time-dependent AUC ...")
  auc_df <- compute_all_timedep_auc(cohorts, models, cfg)
  print(auc_df)

  # ---- Plot ----
  message(">> Drawing AUC comparison figure ...")
  fig <- compose_auc_figure(auc_df)

  out_path <- file.path(save_dir, "DFS_AUC_Comparison.pdf")
  ggsave(out_path, fig, width = cfg$fig_width, height = cfg$fig_height)
  message("  Saved: ", out_path)

  message(">> Done.")
  invisible(list(cindex = cindex_df, auc = auc_df))
}


# ==============================================================================
# 8. CLI Entry
# ==============================================================================
if (!interactive() || identical(environment(), globalenv())) {
  DATA_DIR <- "xxx"
  SAVE_DIR <- "x"

  run_timedep_auc_analysis(DATA_DIR, SAVE_DIR)
}
