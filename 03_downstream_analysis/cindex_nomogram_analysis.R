#!/usr/bin/env Rscript
# ==============================================================================
# cindex_nomogram_analysis.R
# ==============================================================================
# Concordance-index comparison and nomogram for DFS prediction in SCLC.
#
# Pipeline:
#   1. Load & clean cohort data
#   2. Determine optimal risk cut-off (surv_cutpoint on training set)
#   3. Fit three Cox models (AI-only, Clinical-only, Combined)
#   4. Evaluate C-index + pairwise comparisons across all cohorts
#   5. Publication-quality grouped bar plot (NPG palette)
#   6. Nomogram with 1/3/5-year DFS probability axes
#
# Usage:
#   source("cindex_nomogram_analysis.R")
#   run_cindex_analysis(data_dir, save_dir)
# ==============================================================================

# ---- Dependencies ------------------------------------------------------------
suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(ggsci)
  library(survminer)
  library(survival)
  library(rms)
  library(Hmisc)
  library(compareC)
})

# ==============================================================================
# 1. Configuration
# ==============================================================================

default_config <- function() {
  list(
    # CSV file-name -> display-name mapping
    cohort_files = c(
      "CHCAMS_Train" = "CHCAMS_Train.csv",
      "CHCAMS_Val"   = "CHCAMS_Val.csv",
      "HMUCH_Val"    = "HMUCH_Val.csv",
      "TMUGH_Val"    = "TMUGH_Val.csv"
    ),

    # Column names expected in every CSV
    time_col   = "DFS",
    event_col  = "DFSState",
    marker_col = "PreHazard",

    # Factor-coding rules  (raw_value -> label)
    factor_spec = list(
      Gender         = list(levels = c(1, 2),    labels = c("Male", "Female")),
      SmokingHistory = list(levels = c(0, 1),    labels = c("No", "Yes")),
      AJCCStage      = list(levels = c(1, 2, 3), labels = c("Stage I", "Stage II", "Stage III"))
    ),

    # Nomogram time horizons (months)
    nom_times      = c(12, 36, 60),
    nom_fun_labels = c("1-Year DFS Prob", "3-Year DFS Prob", "5-Year DFS Prob"),
    nom_fun_at     = c(0.1, 0.3, 0.5, 0.7, 0.9, 0.95),

    # Bar-plot aesthetics
    bar_ylim  = c(0.4, 1.0),
    bar_width = c(width = 4, height = 3.5)
  )
}


# ==============================================================================
# 2. Data Loading & Cleaning
# ==============================================================================

#' Load all cohort CSVs, drop rows missing AJCCStage.
load_cohorts <- function(data_dir, cfg) {
  lapply(cfg$cohort_files, function(f) {
    read.csv(file.path(data_dir, f), encoding = "utf-8") %>%
      dplyr::filter(!is.na(AJCCStage))
  })
}


#' Apply factor coding + type coercion to a single data.frame.
clean_cohort <- function(df, cfg) {
  for (col_name in names(cfg$factor_spec)) {
    spec <- cfg$factor_spec[[col_name]]
    df[[col_name]] <- factor(df[[col_name]],
                             levels = spec$levels,
                             labels = spec$labels)
  }
  df %>% dplyr::mutate(
    Age       = as.numeric(Age),
    PreHazard = as.numeric(PreHazard),
    DFS       = as.numeric(DFS),
    DFSState  = as.numeric(DFSState)
  )
}


#' Compute the optimal risk-score cut-off from the training cohort and
#' assign High_Risk / Low_Risk labels to every cohort in the list.
assign_risk_groups <- function(cohorts, cfg) {
  train <- cohorts[[1]]
  cp <- surv_cutpoint(train,
                      time      = cfg$time_col,
                      event     = cfg$event_col,
                      variables = cfg$marker_col,
                      progressbar = FALSE)
  cutoff <- cp$cutpoint$cutpoint
  message(sprintf("  Optimal cut-off (PreHazard): %.4f", cutoff))

  lapply(cohorts, function(df) {
    df$HR        <- ifelse(df[[cfg$marker_col]] > cutoff, "High_Risk", "Low_Risk")
    df$Pre_Label <- factor(df$HR, levels = c("High_Risk", "Low_Risk"))
    df
  })
}


# ==============================================================================
# 3. Cox Model Fitting (training set only)
# ==============================================================================

#' Fit three Cox PH models on the training cohort and return them in a list.
#'
#' @return Named list: risk_model, stage_model, combined_model.
fit_cox_models <- function(train_df) {
  dd <- datadist(train_df)
  options(datadist = "dd")

  list(
    risk_model = cph(Surv(DFS, DFSState) ~ PreHazard,
                     data = train_df, x = TRUE, y = TRUE, surv = TRUE),

    stage_model = cph(Surv(DFS, DFSState) ~ AJCCStage + Gender + SmokingHistory + Age,
                      data = train_df, x = TRUE, y = TRUE, surv = TRUE),

    combined_model = cph(Surv(DFS, DFSState) ~ PreHazard + AJCCStage + Gender + SmokingHistory + Age,
                         data = train_df, x = TRUE, y = TRUE, surv = TRUE)
  )
}


# ==============================================================================
# 4. C-index Evaluation
# ==============================================================================

#' Compute C-index + 95 % CI from a linear predictor.
calc_cindex <- function(lp, time, status) {
  x  <- Hmisc::rcorr.cens(-lp, Surv(time, status))
  ci <- x["C Index"]
  se <- x["S.D."] / 2
  c(C = ci, Low = ci - 1.96 * se, High = ci + 1.96 * se)
}


#' Evaluate all three models on one cohort; return a one-row data.frame.
evaluate_cohort <- function(df, cohort_name, models) {
  lp <- lapply(models, function(m) predict(m, newdata = df, type = "lp"))

  res <- lapply(lp, function(l) calc_cindex(l, df$DFS, df$DFSState))

  # Pairwise C-index comparisons (vs. stage_model)
  p_risk_vs_stage <- compareC(df$DFS, df$DFSState,
                              lp$risk_model, lp$stage_model)$pval
  p_comb_vs_stage <- compareC(df$DFS, df$DFSState,
                              lp$combined_model, lp$stage_model)$pval

  data.frame(
    Cohort     = cohort_name,
    Risk_C     = res$risk_model["C"],
    Risk_Low   = res$risk_model["Low"],
    Risk_High  = res$risk_model["High"],
    Stage_C    = res$stage_model["C"],
    Stage_Low  = res$stage_model["Low"],
    Stage_High = res$stage_model["High"],
    Comb_C     = res$combined_model["C"],
    Comb_Low   = res$combined_model["Low"],
    Comb_High  = res$combined_model["High"],
    P_Risk_vs_Stage = p_risk_vs_stage,
    P_Comb_vs_Stage = p_comb_vs_stage,
    row.names  = NULL
  )
}


#' Run evaluate_cohort() over every cohort; return a combined data.frame.
evaluate_all_cohorts <- function(cohorts, models) {
  results <- mapply(evaluate_cohort, cohorts, names(cohorts),
                    MoreArgs = list(models = models), SIMPLIFY = FALSE)
  do.call(rbind, results)
}


# ==============================================================================
# 5. Bar Plot (C-index comparison, NPG palette)
# ==============================================================================

#' Convert p-values to significance stars.
pval_to_star <- function(p) {
  dplyr::case_when(
    p < 0.001 ~ "***",
    p < 0.01  ~ "**",
    p < 0.05  ~ "*",
    TRUE       ~ "ns"
  )
}


#' Reshape evaluation results into long format suitable for ggplot.
prepare_plot_data <- function(results) {
  # Long-form C-index values
  plot_df <- results %>%
    dplyr::select(Cohort, ends_with("_C"), ends_with("_Low"), ends_with("_High")) %>%
    pivot_longer(-Cohort, names_to = "Key", values_to = "Value") %>%
    separate(Key, into = c("Model_Type", "Metric"), sep = "_") %>%
    pivot_wider(names_from = Metric, values_from = Value) %>%
    dplyr::mutate(
      Model_Name = dplyr::case_when(
        Model_Type == "Stage" ~ "Clinical Stage",
        Model_Type == "Risk"  ~ "AI Signature",
        Model_Type == "Comb"  ~ "Combined Model"
      ),
      Model_Name = factor(Model_Name,
                          levels = c("Clinical Stage", "AI Signature", "Combined Model")),
      Cohort = factor(Cohort, levels = unique(results$Cohort))
    )

  # Merge significance stars
  star_df <- results %>%
    dplyr::transmute(
      Cohort,
      Star_Risk = pval_to_star(P_Risk_vs_Stage),
      Star_Comb = pval_to_star(P_Comb_vs_Stage)
    )

  plot_df %>%
    left_join(star_df, by = "Cohort") %>%
    dplyr::mutate(
      Label_Star = dplyr::case_when(
        Model_Type == "Risk" ~ Star_Risk,
        Model_Type == "Comb" ~ Star_Comb,
        TRUE ~ ""
      )
    )
}


#' Draw the grouped bar chart with error bars, value labels, and stars.
plot_cindex_barplot <- function(plot_df, cfg) {
  dodge_w <- 0.8

  ggplot(plot_df, aes(x = Cohort, y = C, fill = Model_Name)) +
    geom_bar(stat = "identity",
             position = position_dodge(dodge_w),
             width = 0.7, color = "black", linewidth = 0.3) +
    geom_errorbar(aes(ymin = Low, ymax = High),
                  position = position_dodge(dodge_w),
                  width = 0.25, linewidth = 0.4) +
    # C-index value inside bar
    geom_text(aes(y = C - 0.025, label = sprintf("%.3f", C)),
              position = position_dodge(dodge_w),
              size = 3.5, color = "white", fontface = "bold") +
    # Significance stars above error bar
    geom_text(aes(y = High + 0.015, label = Label_Star),
              position = position_dodge(dodge_w),
              size = 5, vjust = 0, fontface = "bold") +
    scale_fill_npg() +
    coord_cartesian(ylim = cfg$bar_ylim) +
    scale_y_continuous(breaks = seq(cfg$bar_ylim[1], cfg$bar_ylim[2], 0.1),
                       expand = c(0, 0)) +
    labs(y = "Concordance Index (C-index)", x = NULL, fill = NULL) +
    theme_classic(base_size = 12) +
    theme(
      axis.text.x  = element_text(color = "black", size = 12,
                                   face = "bold", margin = margin(t = 5)),
      axis.text.y  = element_text(color = "black", size = 12),
      axis.title.y = element_text(color = "black", size = 13,
                                   face = "bold", margin = margin(r = 10)),
      axis.line     = element_line(linewidth = 0.6, color = "black"),
      axis.ticks    = element_line(linewidth = 0.6, color = "black"),
      legend.position = "top",
      legend.text   = element_text(size = 11),
      legend.margin = margin(b = 0),
      plot.margin   = margin(20, 20, 10, 10)
    )
}


# ==============================================================================
# 6. Nomogram
# ==============================================================================

#' Save a publication-ready nomogram (base-graphics PDF).
save_nomogram <- function(model, save_path, cfg) {
  surv_obj <- Survival(model)
  funs <- lapply(cfg$nom_times, function(t) {
    function(x) surv_obj(t, lp = x)
  })

  nom <- nomogram(model,
                  fun       = funs,
                  funlabel  = cfg$nom_fun_labels,
                  fun.at    = cfg$nom_fun_at,
                  lp        = FALSE)

  out_file <- file.path(save_path, "Nomogram_Combined_Model.pdf")
  pdf(out_file, width = 5, height = 3.5)
  par(mar = c(2, 2, 2, 2), mgp = c(2.5, 0.8, 0),
      col      = "#00468B",
      col.axis = "black",
      col.lab  = "#ED0000",
      fg       = "#00468B",
      font.lab = 2,
      lwd      = 1.5)
  plot(nom,
       xfrac    = 0.25,
       cex.axis = 0.95,
       cex.var  = 1.15,
       tcl      = -0.4,
       lmgp     = 0.2,
       col.grid = gray(0.85))
  dev.off()
  message("  Nomogram saved: ", out_file)
}


# ==============================================================================
# 7. Main Entry Point
# ==============================================================================

#' Run the full C-index + nomogram pipeline.
#'
#' @param data_dir  Directory containing cohort CSVs.
#' @param save_dir  Directory for output figures.
#' @param cfg       Config list (default: default_config()).
#' @return Invisible evaluation results data.frame.
run_cindex_analysis <- function(data_dir, save_dir, cfg = default_config()) {

  dir.create(save_dir, showWarnings = FALSE, recursive = TRUE)

  # ---- Load & clean ----
  message(">> Loading cohort data ...")
  cohorts_raw   <- load_cohorts(data_dir, cfg)
  cohorts_clean <- lapply(cohorts_raw, clean_cohort, cfg = cfg)

  message(">> Determining optimal risk cut-off ...")
  cohorts_clean <- assign_risk_groups(cohorts_clean, cfg)

  # ---- Fit models on training set ----
  message(">> Fitting Cox models on training cohort ...")
  models <- fit_cox_models(cohorts_clean[[1]])

  # ---- Evaluate across all cohorts ----
  message(">> Evaluating C-index across cohorts ...")
  results <- evaluate_all_cohorts(cohorts_clean, models)
  print(results)

  # ---- Bar plot ----
  message(">> Drawing C-index bar plot ...")
  plot_df <- prepare_plot_data(results)
  p <- plot_cindex_barplot(plot_df, cfg)
  ggsave(file.path(save_dir, "CIndex_Comparison_Barplot.pdf"),
         plot = p,
         width = cfg$bar_width["width"],
         height = cfg$bar_width["height"])
  message("  Saved CIndex_Comparison_Barplot.pdf")

  # ---- Nomogram ----
  message(">> Generating nomogram ...")
  save_nomogram(models$combined_model, save_dir, cfg)

  message(">> All done.")
  invisible(results)
}


# ==============================================================================
# 8. CLI Entry
# ==============================================================================
if (!interactive() || identical(environment(), globalenv())) {
  DATA_DIR <- "xx"
  SAVE_DIR <- "xx"

  run_cindex_analysis(DATA_DIR, SAVE_DIR)
}
