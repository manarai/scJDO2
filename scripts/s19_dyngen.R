#!/usr/bin/env Rscript

options(stringsAsFactors = FALSE)

parse_args <- function(args) {
  out <- list(
    backbone = NULL,
    seed = NULL,
    out = NULL,
    num_cells = 5000L,
    num_simulations = 32L
  )
  i <- 1L
  while (i <= length(args)) {
    key <- args[[i]]
    if (key %in% c("--backbone", "--seed", "--out", "--num-cells", "--num-simulations")) {
      if (i == length(args)) stop("missing value for ", key)
      value <- args[[i + 1L]]
      if (key == "--backbone") out$backbone <- value
      if (key == "--seed") out$seed <- as.integer(value)
      if (key == "--out") out$out <- value
      if (key == "--num-cells") out$num_cells <- as.integer(value)
      if (key == "--num-simulations") out$num_simulations <- as.integer(value)
      i <- i + 2L
    } else {
      stop("unknown argument: ", key)
    }
  }
  if (is.null(out$backbone) || !out$backbone %in% c("bifurcating", "cycle", "linear")) {
    stop("--backbone must be bifurcating, cycle, or linear")
  }
  if (is.null(out$seed) || is.null(out$out)) stop("--seed and --out are required")
  out
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
dir.create(args$out, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages(library(dyngen))
suppressPackageStartupMessages(library(Matrix))
suppressPackageStartupMessages(library(jsonlite))

set.seed(args$seed)
backbone <- switch(
  args$backbone,
  bifurcating = backbone_bifurcating(),
  cycle = backbone_cycle(),
  linear = backbone_linear()
)

# The task fixes these counts.  dyngen's feature count is num_tfs +
# num_targets + num_hks, independent of the backbone's module count.
num_tfs <- 50L
num_targets <- 900L
num_hks <- 50L
num_cells <- args$num_cells

simulation_params <- simulation_default(
  experiment_params = simulation_type_wild_type(num_simulations = args$num_simulations),
  compute_rna_velocity = TRUE
)
experiment_params <- experiment_synchronised(num_timepoints = 3L)

model <- initialise_model(
  backbone = backbone,
  num_cells = num_cells,
  num_tfs = num_tfs,
  num_targets = num_targets,
  num_hks = num_hks,
  distance_metric = "pearson",
  experiment_params = experiment_params,
  simulation_params = simulation_params,
  verbose = TRUE,
  num_cores = 1L,
  id = paste0("s19_", args$backbone, "_s", args$seed)
)

timings <- list()
run_step <- function(label, fun, value) {
  started <- proc.time()[["elapsed"]]
  value <- fun(value)
  timings[[label]] <<- proc.time()[["elapsed"]] - started
  value
}

model <- run_step("generate_tf_network", generate_tf_network, model)
model <- run_step("generate_feature_network", generate_feature_network, model)
model <- run_step("generate_kinetics", generate_kinetics, model)
model <- run_step("generate_gold_standard", generate_gold_standard, model)
model <- run_step("generate_cells", generate_cells, model)
model <- run_step("generate_experiment", generate_experiment, model)

# The list wrapper preserves the raw spliced/unspliced matrices and the
# dyngen metadata without introducing an AnnData dependency into R.
dataset <- wrap_dataset(
  model,
  format = "list",
  store_dimred = FALSE,
  store_cellwise_grn = FALSE,
  store_rna_velocity = TRUE
)

if (nrow(dataset$counts_spliced) != num_cells || ncol(dataset$counts_spliced) != (num_tfs + num_targets + num_hks)) {
  stop("unexpected spliced count shape: ", paste(dim(dataset$counts_spliced), collapse = "x"))
}

write_matrix <- function(x, filename) {
  Matrix::writeMM(x, file.path(args$out, filename))
}
write_matrix(dataset$counts_spliced, "spliced.mtx")
write_matrix(dataset$counts_unspliced, "unspliced.mtx")
if (!is.null(dataset$rna_velocity)) write_matrix(dataset$rna_velocity, "truth_rna_velocity.mtx")

cell_ids <- as.character(dataset$cell_ids)
feature_ids <- as.character(dataset$feature_ids)
write.table(data.frame(cell_id = cell_ids), file.path(args$out, "cells.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)

raw_cell_info <- as.data.frame(model$experiment$cell_info)
raw_cell_info <- raw_cell_info[match(cell_ids, raw_cell_info$cell_id), , drop = FALSE]
if (anyNA(raw_cell_info$cell_id)) stop("cell metadata could not be aligned to counts")

progressions <- as.data.frame(dataset$progressions)
progressions <- progressions[match(cell_ids, progressions$cell_id), , drop = FALSE]
if (anyNA(progressions$cell_id)) stop("progressions could not be aligned to counts")

obs <- data.frame(
  day = as.integer(raw_cell_info$timepoint_group) + 1L,
  truth_simulation_i = as.integer(raw_cell_info$simulation_i),
  truth_simulation_time = as.numeric(raw_cell_info$sim_time),
  truth_backbone_from = as.character(progressions$from),
  truth_backbone_to = as.character(progressions$to),
  truth_backbone_percentage = as.numeric(progressions$percentage),
  stringsAsFactors = FALSE,
  check.names = FALSE
)

if (args$backbone == "bifurcating") {
  obs$truth_branch <- ifelse(
    obs$truth_backbone_to %in% c("sC", "sEndC"), "C",
    ifelse(obs$truth_backbone_to %in% c("sD", "sEndD"), "D", "pre_branch")
  )
} else {
  obs$truth_branch <- args$backbone
}

milestone_percentages <- as.data.frame(dataset$milestone_percentages)
for (milestone_id in unique(as.character(milestone_percentages$milestone_id))) {
  values <- milestone_percentages[milestone_percentages$milestone_id == milestone_id, c("cell_id", "percentage"), drop = FALSE]
  values <- values[match(cell_ids, values$cell_id), , drop = FALSE]
  obs[[paste0("truth_milestone_percentage_", milestone_id)]] <- as.numeric(values$percentage)
}
write.table(obs, file.path(args$out, "obs.tsv"), sep = "\t", quote = FALSE, row.names = FALSE, na = "NA")

feature_info <- as.data.frame(dataset$feature_info)
feature_info <- feature_info[match(feature_ids, feature_info$feature_id), , drop = FALSE]
write.table(feature_info, file.path(args$out, "var.tsv"), sep = "\t", quote = FALSE, row.names = FALSE, na = "NA")

feature_network <- as.data.frame(model$feature_network)
write.table(feature_network, file.path(args$out, "truth_grn.tsv"), sep = "\t", quote = FALSE, row.names = FALSE, na = "NA")
milestone_network <- as.data.frame(model$gold_standard$network)
write.table(milestone_network, file.path(args$out, "truth_milestone_network.tsv"), sep = "\t", quote = FALSE, row.names = FALSE, na = "NA")

settings_core <- list(
  task = "orient/S19",
  dyngen_version = as.character(packageVersion("dyngen")),
  R_version = as.character(getRversion()),
  backbone = args$backbone,
  seed = args$seed,
  num_cells = num_cells,
  num_tfs = num_tfs,
  num_targets = num_targets,
  num_hks = num_hks,
  num_features = as.integer(model$numbers$num_features),
  num_modules = as.integer(model$numbers$num_modules),
  num_simulations = args$num_simulations,
  experiment = list(type = "synchronised", num_timepoints = 3L, pct_between = experiment_params$pct_between),
  compute_rna_velocity = isTRUE(model$simulation_params$compute_rna_velocity),
  compute_dimred = isTRUE(model$simulation_params$compute_dimred),
  store_reaction_firings = isTRUE(model$simulation_params$store_reaction_firings),
  store_reaction_propensities = isTRUE(model$simulation_params$store_reaction_propensities),
  compute_cellwise_grn = isTRUE(model$simulation_params$compute_cellwise_grn),
  distance_metric = model$distance_metric,
  timings_seconds = timings
)
write_json(settings_core, file.path(args$out, "settings.json"), auto_unbox = TRUE, pretty = TRUE)

settings_text <- c(
  "# Full dyngen settings and provenance",
  paste0("# generated_utc: ", format(Sys.time(), tz = "UTC", usetz = TRUE)),
  capture.output(print(settings_core)),
  "",
  "## numbers",
  capture.output(print(model$numbers)),
  "## tf_network_params",
  capture.output(print(model$tf_network_params)),
  "## feature_network_params",
  capture.output(print(model$feature_network_params)),
  "## kinetics_params",
  capture.output(print(model$kinetics_params)),
  "## gold_standard_params",
  capture.output(print(model$gold_standard_params)),
  "## simulation_params",
  capture.output(print(model$simulation_params)),
  "## experiment_params",
  capture.output(print(model$experiment_params)),
  "## sessionInfo",
  capture.output(sessionInfo())
)
writeLines(settings_text, file.path(args$out, "settings.txt"))

saveRDS(model, file.path(args$out, "model.rds"), compress = "xz")
write_json(list(
  status = "generated",
  task = "orient/S19",
  backbone = args$backbone,
  seed = args$seed,
  cells = nrow(dataset$counts_spliced),
  genes = ncol(dataset$counts_spliced),
  generated_utc = format(Sys.time(), tz = "UTC", usetz = TRUE)
), file.path(args$out, "checkpoint.json"), auto_unbox = TRUE, pretty = TRUE)

cat(jsonlite::toJSON(settings_core, auto_unbox = TRUE, pretty = TRUE), "\n")
