library(piecewiseSEM)
library(tidyverse)
library(ggplot2)
library(patchwork)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 2) stop("Usage: Rscript sem/run_hcr_sem_kfjp_coverage_breadth.R [input.csv] [output_directory]")
base <- Sys.getenv("KFJP_SEM_BASE", unset = file.path("sem", "data"))
source_data <- if (length(args) >= 1) args[[1]] else file.path(base, "DATA", "KFJP_superiority_tests", "kfjp_indicator_test_dataset.csv")
out_dir <- if (length(args) >= 2) args[[2]] else file.path("sem", "outputs")
if (!file.exists(source_data)) stop("Input CSV not found: ", source_data, ". Supply the analysis CSV as the first argument or set KFJP_SEM_BASE.")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

sem_data <- read_csv(source_data, show_col_types = FALSE) %>%
  mutate(
    subject = factor(subject),
    subject_bio = if_else(subject == "bio", 1, 0),
    subject_che = if_else(subject == "che", 1, 0),
    subject_mat = if_else(subject == "mat", 1, 0)
  ) %>%
  drop_na(
    citation_perf, productivity, career_age, breadth, concentration,
    q1h_non_kfjp_pub, kfjp_pub, kfjp_surplus, kfjp_coverage_breadth
  )

coverage_model <- psem(
  lm(q1h_non_kfjp_pub ~ productivity + career_age + breadth + concentration + subject_bio + subject_che + subject_mat, data = sem_data),
  lm(kfjp_pub ~ kfjp_coverage_breadth + productivity + career_age + breadth + concentration + subject_bio + subject_che + subject_mat, data = sem_data),
  lm(kfjp_surplus ~ kfjp_pub + q1h_non_kfjp_pub + kfjp_coverage_breadth + productivity + career_age + breadth + concentration + subject_bio + subject_che + subject_mat, data = sem_data),
  lm(citation_perf ~ kfjp_surplus + kfjp_pub + q1h_non_kfjp_pub + kfjp_coverage_breadth + productivity + career_age + breadth + concentration + subject_bio + subject_che + subject_mat, data = sem_data),
  q1h_non_kfjp_pub %~~% kfjp_pub,
  data = as.data.frame(sem_data)
)

sink(file.path(out_dir, "hcr_sem_kfjp_coverage_breadth_summary.txt"))
cat("HCR SEM: KFJP coverage breadth model\n")
cat("This model extends the accepted full-record allocation SEM by adding KFJP coverage breadth.\n")
cat("KFJP coverage breadth = log(1 + number of KFJP journals in which the scholar published).\n")
cat("N =", nrow(sem_data), "\n\n")
print(summary(coverage_model))
cat("\nFisher C:\n")
print(fisherC(coverage_model))
cat("\nAIC:\n")
print(AIC(coverage_model))
cat("\ndSep:\n")
print(dSep(coverage_model))
cat("\nCoefficients:\n")
print(coefs(coverage_model))
sink()

path_coeffs <- coefs(coverage_model)
write_csv(path_coeffs, file.path(out_dir, "hcr_sem_kfjp_coverage_breadth_path_coefficients.csv"))

filtered_path_coeffs <- path_coeffs[!grepl("^~~", path_coeffs$Predictor), ] %>%
  as_tibble(.name_repair = "unique") %>%
  filter(!str_starts(Predictor, "subject"))

find_paths <- function(predictor, data, current_path = integer(0)) {
  responses <- which(data$Predictor == predictor)
  if (length(responses) == 0) {
    return(list(current_path))
  }

  paths <- list()
  for (response in responses) {
    response_node <- data$Response[response]
    response_paths <- find_paths(response_node, data, c(current_path, response))
    paths <- c(paths, response_paths)
  }
  return(paths)
}

calc_effectsize <- function(data, target = NA) {
  res <- tibble()
  for (pre in unique(data$Predictor)) {
    paths <- find_paths(pre, data)
    for (path in paths) {
      path <- unlist(path)
      while (!is.na(target) & !is_empty(path) & !data$Response[last(path)] %in% target) {
        path <- path[-length(path)]
      }
      if (is_empty(path)) {
        next
      }
      response <- data$Response[last(path)]
      effectsize <- data$Std.Estimate[path] %>% prod(., na.rm = TRUE)
      res <- bind_rows(res, tibble(Predictor = pre, Response = response, EffectSize = effectsize))
    }
  }
  res %>%
    unique() %>%
    summarise(EffectSize = sum(EffectSize, na.rm = TRUE), .by = c("Predictor", "Response"))
}

net_effect <- calc_effectsize(filtered_path_coeffs, target = "citation_perf") %>%
  filter(Response == "citation_perf") %>%
  mutate(type = case_when(
    Predictor %in% c("productivity", "career_age") ~ "Academic accumulation",
    Predictor %in% c("breadth", "concentration") ~ "General portfolio structure",
    Predictor %in% c("q1h_non_kfjp_pub") ~ "Same-tier non-KFJP allocation",
    Predictor %in% c("kfjp_pub") ~ "KFJP allocation",
    Predictor %in% c("kfjp_coverage_breadth") ~ "KFJP coverage breadth",
    Predictor %in% c("kfjp_surplus") ~ "KFJP citation surplus",
    TRUE ~ "Other"
  ))
write_csv(net_effect, file.path(out_dir, "hcr_sem_kfjp_coverage_breadth_net_effect.csv"))

label_map <- c(
  productivity = "Publication\nvolume",
  career_age = "Academic\nage",
  breadth = "Overall\njournal\nbreadth",
  concentration = "Portfolio\nconcentration",
  q1h_non_kfjp_pub = "Q1H non-KFJP\npublication\nshare",
  kfjp_coverage_breadth = "KFJP\ncoverage\nbreadth",
  kfjp_pub = "KFJP\npublication\nshare",
  kfjp_surplus = "KFJP citation\nsurplus",
  citation_perf = "Citation\nperformance"
)

edge_data <- filtered_path_coeffs %>%
  filter(Response %in% c("q1h_non_kfjp_pub", "kfjp_pub", "kfjp_surplus", "citation_perf")) %>%
  filter(Predictor %in% names(label_map)) %>%
  filter(P.Value < 0.05) %>%
  mutate(
    sign = if_else(Std.Estimate >= 0, "positive", "negative"),
    edge_label = sprintf("%.2f%s", Std.Estimate, if_else(P.Value < 0.05, "*", ""))
  )

nodes <- tribble(
  ~name, ~x, ~y, ~kind,
  "career_age", 0.0, 4.4, "control",
  "productivity", 0.0, 3.3, "control",
  "breadth", 0.0, 2.2, "portfolio",
  "concentration", 0.0, 1.1, "portfolio",
  "q1h_non_kfjp_pub", 2.15, 4.0, "alternative",
  "kfjp_coverage_breadth", 2.15, 2.65, "coverage",
  "kfjp_pub", 2.15, 1.3, "kfjp",
  "kfjp_surplus", 4.35, 2.65, "advantage",
  "citation_perf", 6.35, 2.65, "outcome"
) %>%
  mutate(label = unname(label_map[name]))

plot_edges <- edge_data %>%
  left_join(nodes %>% select(Predictor = name, x0 = x, y0 = y), by = "Predictor") %>%
  left_join(nodes %>% select(Response = name, x1 = x, y1 = y), by = "Response") %>%
  mutate(
    x_start = x0 + if_else(x1 >= x0, 0.38, -0.38),
    x_end = x1 - if_else(x1 >= x0, 0.48, -0.48),
    label_x = (x_start + x_end) / 2,
    label_y = (y0 + y1) / 2
  )

p_sem <- ggplot() +
  geom_segment(
    data = plot_edges,
    aes(x = x_start, y = y0, xend = x_end, yend = y1, color = sign, linewidth = abs(Std.Estimate)),
    arrow = arrow(length = unit(0.16, "cm"), type = "closed"),
    lineend = "round"
  ) +
  geom_label(
    data = nodes,
    aes(x = x, y = y, label = label, fill = kind),
    size = 2.65, label.size = 0.25, label.r = unit(0.06, "lines"), color = "black"
  ) +
  geom_label(
    data = plot_edges,
    aes(x = label_x, y = label_y, label = edge_label),
    size = 2.45, label.size = 0, fill = "white", alpha = 0.88
  ) +
  scale_color_manual(values = c(positive = "black", negative = "#C00000"), guide = "none") +
  scale_linewidth(range = c(0.25, 1.4), guide = "none") +
  scale_fill_manual(values = c(
    control = "#D9EAF7",
    portfolio = "#FCE4D6",
    alternative = "#E7E6E6",
    coverage = "#D9EAD3",
    kfjp = "#E2F0D9",
    advantage = "#C6E0B4",
    outcome = "#FCE4D6"
  ), guide = "none") +
  coord_cartesian(xlim = c(-0.75, 7.0), ylim = c(0.55, 4.75), clip = "off") +
  theme_void() +
  ggtitle("(a) KFJP coverage-breadth SEM")

net_effect_plot <- net_effect %>%
  mutate(Predictor = factor(
    Predictor,
    levels = c("career_age", "productivity", "breadth", "concentration", "q1h_non_kfjp_pub", "kfjp_pub", "kfjp_coverage_breadth", "kfjp_surplus"),
    labels = c(
      "Academic age", "Publication volume", "Overall journal breadth", "Portfolio concentration",
      "Q1H non-KFJP publication share", "KFJP publication share", "KFJP coverage breadth", "KFJP citation surplus"
    )
  ))

p_net <- ggplot(net_effect_plot, aes(x = Predictor, y = EffectSize, fill = type)) +
  geom_hline(yintercept = 0, color = "black", lty = 2, linewidth = 0.2) +
  geom_col(width = 0.58) +
  scale_fill_manual(values = c(
    "Academic accumulation" = "#8FAADC",
    "General portfolio structure" = "#F4B183",
    "Same-tier non-KFJP allocation" = "#BFBFBF",
    "KFJP allocation" = "#A9D18E",
    "KFJP coverage breadth" = "#548235",
    "KFJP citation surplus" = "#70AD47",
    "Other" = "grey70"
  )) +
  coord_flip() +
  labs(x = NULL, y = "Net standardized effect on citation performance", fill = NULL) +
  theme_bw(base_size = 11) +
  theme(panel.grid = element_blank(), legend.position = "none")

ggsave(file.path(out_dir, "hcr_sem_kfjp_coverage_breadth_net_effect.png"), p_net, width = 7, height = 4.5, dpi = 300)
ggsave(file.path(out_dir, "hcr_sem_kfjp_coverage_breadth_net_effect.pdf"), p_net, width = 7, height = 4.5)

p_combined <- p_sem + p_net + plot_layout(widths = c(1.55, 1))
ggsave(file.path(out_dir, "hcr_sem_kfjp_coverage_breadth_combined.png"), p_combined, width = 12.4, height = 5.6, dpi = 300)
ggsave(file.path(out_dir, "hcr_sem_kfjp_coverage_breadth_combined.pdf"), p_combined, width = 12.4, height = 5.6)

first_layer_nodes <- c("career_age", "productivity", "breadth", "concentration")

edge_data_mechanism <- edge_data %>%
  filter(!(Predictor %in% first_layer_nodes & Response == "citation_perf"))

nodes_mechanism <- tribble(
  ~name, ~x, ~y, ~kind,
  "career_age", 0.0, 4.55, "control",
  "productivity", 0.0, 3.45, "control",
  "breadth", 0.0, 2.35, "portfolio",
  "concentration", 0.0, 1.25, "portfolio",
  "q1h_non_kfjp_pub", 2.2, 4.25, "alternative",
  "kfjp_coverage_breadth", 2.35, 3.15, "coverage",
  "kfjp_pub", 2.2, 1.35, "kfjp",
  "kfjp_surplus", 4.45, 2.25, "advantage",
  "citation_perf", 6.5, 2.8, "outcome"
) %>%
  mutate(label = unname(label_map[name]))

plot_edges_mechanism <- edge_data_mechanism %>%
  left_join(nodes_mechanism %>% select(Predictor = name, x0 = x, y0 = y), by = "Predictor") %>%
  left_join(nodes_mechanism %>% select(Response = name, x1 = x, y1 = y), by = "Response") %>%
  mutate(
    x_start = x0 + if_else(x1 >= x0, 0.38, -0.38),
    x_end = x1 - if_else(x1 >= x0, 0.48, -0.48),
    label_x = (x_start + x_end) / 2,
    label_y = (y0 + y1) / 2
  )

p_sem_mechanism <- ggplot() +
  geom_segment(
    data = plot_edges_mechanism,
    aes(x = x_start, y = y0, xend = x_end, yend = y1, color = sign, linewidth = abs(Std.Estimate)),
    arrow = arrow(length = unit(0.16, "cm"), type = "closed"),
    lineend = "round"
  ) +
  geom_label(
    data = nodes_mechanism,
    aes(x = x, y = y, label = label, fill = kind),
    size = 2.65, label.size = 0.25, label.r = unit(0.06, "lines"), color = "black"
  ) +
  geom_label(
    data = plot_edges_mechanism,
    aes(x = label_x, y = label_y, label = edge_label),
    size = 2.45, label.size = 0, fill = "white", alpha = 0.88
  ) +
  scale_color_manual(values = c(positive = "black", negative = "#C00000"), guide = "none") +
  scale_linewidth(range = c(0.25, 1.4), guide = "none") +
  scale_fill_manual(values = c(
    control = "#D9EAF7",
    portfolio = "#FCE4D6",
    alternative = "#E7E6E6",
    coverage = "#D9EAD3",
    kfjp = "#E2F0D9",
    advantage = "#C6E0B4",
    outcome = "#FCE4D6"
  ), guide = "none") +
  coord_cartesian(xlim = c(-0.75, 7.05), ylim = c(0.7, 4.9), clip = "off") +
  theme_void() +
  theme(
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA)
  ) +
  ggtitle("(a) KFJP coverage-breadth SEM: mechanism display")

p_combined_mechanism <- p_sem_mechanism + p_net + plot_layout(widths = c(1.55, 1))
ggsave(file.path(out_dir, "hcr_sem_kfjp_coverage_breadth_mechanism_display.png"), p_sem_mechanism, width = 7.8, height = 5.6, dpi = 300, bg = "white")
ggsave(file.path(out_dir, "hcr_sem_kfjp_coverage_breadth_mechanism_display.pdf"), p_sem_mechanism, width = 7.8, height = 5.6, bg = "white")
ggsave(file.path(out_dir, "hcr_sem_kfjp_coverage_breadth_mechanism_display_combined.png"), p_combined_mechanism, width = 12.4, height = 5.6, dpi = 300, bg = "white")
ggsave(file.path(out_dir, "hcr_sem_kfjp_coverage_breadth_mechanism_display_combined.pdf"), p_combined_mechanism, width = 12.4, height = 5.6, bg = "white")

log_text <- c(
  "# KFJP Coverage-Breadth SEM Log",
  "",
  "## Purpose",
  "",
  "This model extends the accepted full-record parallel-allocation SEM by adding KFJP coverage breadth.",
  "",
  "## Indicator Definition",
  "",
  "`kfjp_coverage_breadth = log(1 + number of KFJP journals in which a scholar has published)`.",
  "",
  "## Model Logic",
  "",
  "```text",
  "Academic age / publication volume / overall journal breadth / portfolio concentration",
  "        -> Q1H non-KFJP publication share",
  "        -> KFJP publication share",
  "",
  "Q1H non-KFJP publication share / KFJP coverage breadth / KFJP publication share",
  "        -> KFJP citation surplus",
  "        -> Citation performance",
  "```",
  "",
  "## Key Interpretation",
  "",
  "The model tests whether broader coverage of KFJP journals is associated with higher citation performance after accounting for KFJP publication share, Q1H non-KFJP publication share, KFJP citation surplus, academic accumulation, general portfolio structure, and field differences.",
  "",
  "## Caution",
  "",
  "KFJP coverage breadth is structurally related to KFJP publication share. The result supports a portfolio-coverage association, not a causal claim.",
  "",
  "## Additional Figure",
  "",
  "`hcr_sem_kfjp_coverage_breadth_mechanism_display.*` removes the displayed direct paths from first-layer controls to citation performance and slightly repositions the four portfolio/allocation indicators. This is a visualization-only version; the statistical model and coefficients are unchanged."
)
writeLines(log_text, file.path(out_dir, "SEM_kfjp_coverage_breadth_model_log.md"))

cat("Saved KFJP coverage-breadth SEM outputs to:", out_dir, "\n")
