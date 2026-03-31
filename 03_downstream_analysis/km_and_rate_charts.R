#!/usr/bin/env Rscript
# ==============================================================================
# km_and_rate_charts.R
# ==============================================================================
# Kaplan-Meier survival curves (with risk table, HR annotation, vertical
# reference lines at 1/3/5 years) and cumulative event-rate bar charts
# for high- vs low-risk groups across multiple SCLC cohorts.
#
# Output:
#   - KM_Survival_Curves.pdf   (8 panels: 4 cohorts x DFS/OS)
#   - Cumulative_Rate_Bars.pdf (8 panels: 4 cohorts x Recurrence/Death)
#
# Usage:
#   source("km_and_rate_charts.R")
#   run_km_analysis(data_dir, save_dir)
# ==============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(survival)
  library(survminer)
  library(ggplot2)
  library(ggpubr)
  library(scales)
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

    marker_col = "PreHazard",

    # Two endpoints to analyse (each needs time + status columns)
    endpoints = list(
      DFS = list(time = "DFS",  status = "DFSState",
                 surv_ylab = "Disease-Free Survival",
                 bar_ylab  = "Recurrence Rate"),
      OS  = list(time = "OS",   status = "OSState",
                 surv_ylab = "Overall Survival",
                 bar_ylab  = "Death Rate")
    ),

    # Risk-group display
    palette      = c("#A1C5DE", "#F4B981"),
    legend_labs  = c("SD/PD", "CR/PR"),

    # Landmark time points (months)
    time_points = c(12, 36, 60),
    time_labels = c("1-Year", "3-Year", "5-Year"),

    # Typography (base font size in pt)
    base_font = 8,

    # Export dimensions (cm)
    km_size  = c(width = 24, height = 15),
    bar_size = c(width = 20, height = 12)
  )
}


# ==============================================================================
# 2. Data Helpers
# ==============================================================================

load_cohorts <- function(data_dir, cfg) {
  lapply(cfg$cohort_files, function(f) {
    read.csv(file.path(data_dir, f), encoding = "utf-8")
  })
}

assign_risk_groups <- function(cohorts, cfg) {
  cp <- surv_cutpoint(cohorts[[1]],
                      time = "DFS", event = "DFSState",
                      variables = cfg$marker_col, progressbar = FALSE)
  cutoff <- cp$cutpoint$cutpoint
  message(sprintf("  Risk cut-off: %.4f", cutoff))

  lapply(cohorts, function(df) {
    df$HR        <- ifelse(df[[cfg$marker_col]] > cutoff, "High_Risk", "Low_Risk")
    df$Pre_Label <- factor(df$HR, levels = c("Low_Risk", "High_Risk"))
    df
  })
}


# ==============================================================================
# 3. Kaplan-Meier Plot
# ==============================================================================

#' Build a single KM panel (curve + risk table + HR annotation).
#'
#' @param df      Data with Pre_Label column.
#' @param time    Name of time column.
#' @param status  Name of event-status column.
#' @param ylab    Y-axis label.
#' @param title   Panel title.
#' @param cfg     Config list.
#' @return A ggsurvplot object.
build_km_plot <- function(df, time, status, ylab, title, cfg) {
  bfs <- cfg$base_font
  text_mm   <- bfs / 2.83464567
  censor_mm <- 0.7 * text_mm

  scale_factor <- 0.75
  curve_mm <- (1.5 / scale_factor) * 0.3527778
  axis_mm  <- (0.5 / scale_factor) * 0.3527778

  # Temporarily map columns to standard names for survfit
  df$.time   <- df[[time]]
  df$.status <- df[[status]]

  fit <- survfit(Surv(.time, .status) ~ Pre_Label, data = df)

  # Cox HR text
  cox_fit <- coxph(Surv(.time, .status) ~ Pre_Label, data = df)
  ci <- summary(cox_fit)$conf.int
  hr_txt <- sprintf("HR=%.3f (%.3f\u2013%.3f)", ci[1], ci[3], ci[4])

  p0 <- ggsurvplot(
    fit, data = df, title = title,
    risk.table = TRUE, risk.table.col = "strata",
    palette = cfg$palette,
    pval = TRUE, pval.method = TRUE,
    pval.size = text_mm, pval.method.size = text_mm,
    ylab = ylab, xlab = "Time (Months)",
    tables.height = 0.25,
    font.x = c(bfs, "plain", "black"),
    font.y = c(bfs, "plain", "black"),
    font.tickslab = c(bfs, "plain", "black"),
    ylim = c(0, 1), size = curve_mm,
    legend.labs = cfg$legend_labs,
    risk.table.title = "Number at risk",
    conf.int = TRUE,
    ggtheme = theme_classic(base_size = bfs)
  )

  # --- Main plot tweaks ---
  p0$plot <- p0$plot +
    theme(
      panel.grid     = element_blank(),
      plot.title     = element_text(size = bfs),
      axis.title     = element_text(size = bfs),
      axis.text      = element_text(size = bfs),
      legend.title   = element_text(size = bfs),
      legend.text    = element_text(size = bfs)
    ) +
    geom_vline(xintercept = cfg$time_points,
               linetype = "dashed", color = "grey60", linewidth = axis_mm) +
    annotate("text", x = 0, y = 0.12, label = hr_txt,
             size = text_mm, hjust = 0)

  # --- Risk table tweaks ---
  p0$table <- p0$table +
    theme_classic(base_size = bfs * 0.8, base_line_size = 0.5) +
    theme(
      plot.title = element_text(size = bfs),
      axis.text  = element_text(size = bfs),
      axis.title = element_text(size = bfs),
      axis.line  = element_line(linewidth = axis_mm),
      axis.ticks = element_line(linewidth = axis_mm)
    )

  # Force consistent text / censor sizes across layers
  for (i in seq_along(p0$table$layers)) {
    if (inherits(p0$table$layers[[i]]$geom, "GeomText"))
      p0$table$layers[[i]]$aes_params$size <- text_mm
  }
  for (i in seq_along(p0$plot$layers)) {
    if (inherits(p0$plot$layers[[i]]$geom, "GeomPoint"))
      p0$plot$layers[[i]]$aes_params$size <- censor_mm * (8 / 6)
  }

  p0
}


# ==============================================================================
# 4. Cumulative Event-Rate Bar Chart
# ==============================================================================

#' Compute truncated log-rank p-value at a single landmark time.
truncated_logrank_p <- function(df, time_col, status_col, t_month) {
  df$t_time   <- pmin(df[[time_col]], t_month)
  df$t_status <- ifelse(df[[time_col]] > t_month, 0, df[[status_col]])

  sdf <- tryCatch(
    survdiff(Surv(t_time, t_status) ~ Pre_Label, data = df),
    error = function(e) NULL
  )
  if (is.null(sdf)) return("NA")
  p <- 1 - pchisq(sdf$chisq, length(sdf$n) - 1)
  if (p < 0.001) "P < 0.001" else sprintf("P = %.3f", p)
}


#' Build a cumulative-rate bar chart for one cohort + one endpoint.
build_rate_bar <- function(df, time_col, status_col, title, ylab, cfg) {
  bfs     <- cfg$base_font
  text_mm <- bfs / 2.83464567
  axis_mm <- (0.5 / 0.75) * 0.3527778

  # KM estimates at landmark times
  form <- as.formula(paste0("Surv(", time_col, ", ", status_col, ") ~ Pre_Label"))
  fit  <- survfit(form, data = df)
  ss   <- summary(fit, times = cfg$time_points, extend = TRUE)

  plot_df <- data.frame(
    TimePoint = factor(rep(cfg$time_labels, times = length(levels(df$Pre_Label))),
                       levels = cfg$time_labels),
    Group     = factor(gsub("Pre_Label=", "", as.character(ss$strata)),
                       levels = c("High_Risk", "Low_Risk")),
    Rate      = 1 - ss$surv,
    Lower     = pmax(1 - ss$upper, 0),
    Upper     = pmin(1 - ss$lower, 1)
  )

  # Per-timepoint p-values
  p_texts <- vapply(cfg$time_points, function(t)
    truncated_logrank_p(df, time_col, status_col, t), character(1))

  anno_df <- data.frame(
    TimePoint = factor(cfg$time_labels, levels = cfg$time_labels),
    Label     = p_texts,
    Y_Pos     = vapply(cfg$time_labels, function(tl)
      max(plot_df$Upper[plot_df$TimePoint == tl], 0.5, na.rm = TRUE) + 0.08,
      numeric(1))
  )

  ggplot(plot_df, aes(x = TimePoint, y = Rate, fill = Group)) +
    geom_bar(stat = "identity", position = position_dodge(0.8),
             width = 0.7, color = "black", linewidth = 0.2) +
    geom_errorbar(aes(ymin = Lower, ymax = Upper),
                  position = position_dodge(0.8), width = 0.25, linewidth = 0.3) +
    geom_text(aes(label = paste0(round(Rate * 100, 1), "%")),
              position = position_dodge(0.8), vjust = -0.5,
              size = text_mm * 0.8) +
    geom_text(data = anno_df,
              aes(x = TimePoint, y = Y_Pos, label = Label),
              inherit.aes = FALSE, size = text_mm,
              fontface = "italic") +
    scale_fill_manual(values = c("High_Risk" = "#A1C5DE", "Low_Risk" = "#F4B981"),
                      labels = c("High Risk", "Low Risk")) +
    scale_y_continuous(labels = percent, limits = c(0, 1.25), expand = c(0, 0)) +
    labs(title = title, x = NULL, y = ylab, fill = NULL) +
    theme_classic(base_size = bfs) +
    theme(
      plot.title  = element_text(hjust = 0.5, size = bfs, face = "bold"),
      axis.text   = element_text(color = "black", size = bfs),
      axis.title  = element_text(color = "black", size = bfs),
      legend.position  = "top",
      legend.text      = element_text(size = bfs),
      legend.key.size  = unit(0.4, "cm"),
      axis.line   = element_line(linewidth = axis_mm),
      axis.ticks  = element_line(linewidth = axis_mm)
    )
}


# ==============================================================================
# 5. Batch Generation
# ==============================================================================

#' Generate all KM + bar panels across cohorts and endpoints.
#'
#' @return Named list with $km_plots and $bar_plots (each a list of objects).
generate_all_panels <- function(cohorts, cfg) {
  km_plots  <- list()
  bar_plots <- list()

  for (cname in names(cohorts)) {
    for (ep_key in names(cfg$endpoints)) {
      ep    <- cfg$endpoints[[ep_key]]
      tag   <- paste(cname, ep_key, sep = ": ")

      message("  ", tag)
      km_plots[[tag]]  <- build_km_plot(
        cohorts[[cname]], ep$time, ep$status,
        ylab = ep$surv_ylab, title = tag, cfg = cfg
      )
      bar_plots[[tag]] <- build_rate_bar(
        cohorts[[cname]], ep$time, ep$status,
        title = tag, ylab = ep$bar_ylab, cfg = cfg
      )
    }
  }

  list(km_plots = km_plots, bar_plots = bar_plots)
}


# ==============================================================================
# 6. Main Entry Point
# ==============================================================================

#' Run the full KM + rate-bar pipeline.
#'
#' @param data_dir  Directory with cohort CSVs.
#' @param save_dir  Output directory.
#' @param cfg       Config list.
#' @return Invisible panel lists.
run_km_analysis <- function(data_dir, save_dir, cfg = default_config()) {

  dir.create(save_dir, showWarnings = FALSE, recursive = TRUE)

  message(">> Loading cohorts ...")
  cohorts <- load_cohorts(data_dir, cfg)

  message(">> Assigning risk groups ...")
  cohorts <- assign_risk_groups(cohorts, cfg)

  message(">> Generating panels ...")
  panels <- generate_all_panels(cohorts, cfg)

  # ---- Save KM curves ----
  km_file <- file.path(save_dir, "KM_Survival_Curves.pdf")
  res_km  <- arrange_ggsurvplots(panels$km_plots, print = FALSE,
                                  ncol = 4, nrow = 2)
  ggsave(km_file, res_km,
         width = cfg$km_size["width"], height = cfg$km_size["height"],
         units = "cm")
  message("  Saved: ", km_file)

  # ---- Save bar charts ----
  bar_file <- file.path(save_dir, "Cumulative_Rate_Bars.pdf")
  res_bar  <- ggarrange(plotlist = panels$bar_plots,
                         ncol = 4, nrow = 2, labels = "AUTO")
  ggsave(bar_file, res_bar,
         width = cfg$bar_size["width"], height = cfg$bar_size["height"],
         units = "cm", device = cairo_pdf)
  message("  Saved: ", bar_file)

  message(">> Done.")
  invisible(panels)
}


# ==============================================================================
# 7. CLI Entry
# ==============================================================================
if (!interactive() || identical(environment(), globalenv())) {
  DATA_DIR <- "xxx"
  SAVE_DIR <- "xxxx"

  run_km_analysis(DATA_DIR, SAVE_DIR)
}
