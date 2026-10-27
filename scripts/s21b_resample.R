#!/usr/bin/env Rscript

options(stringsAsFactors = FALSE)

parse_args <- function(args) {
  out <- list(key = NULL, seed = NULL, timepoints = NULL, backbone = NULL, output = NULL)
  i <- 1L
  while (i <= length(args)) {
    key <- args[[i]]
    if (!key %in% c("--key", "--seed", "--timepoints", "--backbone", "--output")) {
      stop("unknown argument: ", key)
    }
    if (i == length(args)) stop("missing value for ", key)
    value <- args[[i + 1L]]
    if (key == "--key") out$key <- value
    if (key == "--seed") out$seed <- as.integer(value)
    if (key == "--timepoints") out$timepoints <- as.integer(value)
    if (key == "--backbone") out$backbone <- value
    if (key == "--output") out$output <- value
    i <- i + 2L
  }
  if (is.null(out$key) || is.null(out$seed) || is.null(out$timepoints) ||
      is.null(out$backbone) || is.null(out$output)) {
    stop("--key, --seed, --timepoints, --backbone, and --output are required")
  }
  if (!out$timepoints %in% c(8L, 12L)) stop("timepoints must be 8 or 12")
  if (!out$backbone %in% c("bifurcating", "cycle", "linear")) stop("invalid backbone")
  out
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
root <- normalizePath(".", mustWork = TRUE)
source_dir <- file.path(root, "benchmarks/dyngen/raw", args$key)
output_dir <- file.path(root, args$output)
if (!file.exists(file.path(source_dir, "model.rds"))) stop("missing saved model: ", source_dir)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages(library(dyngen))
suppressPackageStartupMessages(library(Matrix))
suppressPackageStartupMessages(library(jsonlite))

model <- readRDS(file.path(source_dir, "model.rds"))
model$experiment_params <- experiment_synchronised(num_timepoints = args$timepoints)
set.seed(args$seed)
started <- proc.time()[["elapsed"]]
model <- generate_experiment(model)
generate_experiment_seconds <- proc.time()[["elapsed"]] - started

dataset <- wrap_dataset(
  model,
  format = "list",
  store_dimred = FALSE,
  store_cellwise_grn = FALSE,
  store_rna_velocity = TRUE
)
if (nrow(dataset$counts_spliced) != 5000L || ncol(dataset$counts_spliced) != 1000L) {
  stop("unexpected spliced count shape: ", paste(dim(dataset$counts_spliced), collapse = "x"))
}

write_matrix <- function(x, filename) Matrix::writeMM(x, file.path(output_dir, filename))
write_matrix(dataset$counts_spliced, "spliced.mtx")
write_matrix(dataset$counts_unspliced, "unspliced.mtx")
if (!is.null(dataset$rna_velocity)) write_matrix(dataset$rna_velocity, "truth_rna_velocity.mtx")

cell_ids <- as.character(dataset$cell_ids)
feature_ids <- as.character(dataset$feature_ids)
write.table(data.frame(cell_id = cell_ids), file.path(output_dir, "cells.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
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
  values <- milestone_percentages[milestone_percentages$milestone_id == milestone_id,
                                  c("cell_id", "percentage"), drop = FALSE]
  values <- values[match(cell_ids, values$cell_id), , drop = FALSE]
  obs[[paste0("truth_milestone_percentage_", milestone_id)]] <- as.numeric(values$percentage)
}
write.table(obs, file.path(output_dir, "obs.tsv"), sep = "\t", quote = FALSE,
            row.names = FALSE, na = "NA")

feature_info <- as.data.frame(dataset$feature_info)
feature_info <- feature_info[match(feature_ids, feature_info$feature_id), , drop = FALSE]
write.table(feature_info, file.path(output_dir, "var.tsv"), sep = "\t", quote = FALSE,
            row.names = FALSE, na = "NA")
write.table(as.data.frame(model$feature_network), file.path(output_dir, "truth_grn.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE, na = "NA")
write.table(as.data.frame(model$gold_standard$network),
            file.path(output_dir, "truth_milestone_network.tsv"), sep = "\t",
            quote = FALSE, row.names = FALSE, na = "NA")

settings_core <- list(
  task = "orient/S21b",
  dyngen_version = as.character(packageVersion("dyngen")),
  R_version = as.character(getRversion()),
  source_key = args$key,
  key = paste0(args$key, "_T", args$timepoints),
  backbone = args$backbone,
  seed = args$seed,
  num_cells = as.integer(model$numbers$num_cells),
  num_tfs = as.integer(model$numbers$num_tfs),
  num_targets = as.integer(model$numbers$num_targets),
  num_hks = as.integer(model$numbers$num_hks),
  num_features = as.integer(model$numbers$num_features),
  num_modules = as.integer(model$numbers$num_modules),
  num_simulations = as.integer(nrow(model$simulation_params$experiment_params)),
  experiment = list(type = "synchronised", num_timepoints = args$timepoints,
                    pct_between = model$experiment_params$pct_between),
  compute_rna_velocity = isTRUE(model$simulation_params$compute_rna_velocity),
  compute_dimred = isTRUE(model$simulation_params$compute_dimred),
  store_reaction_firings = isTRUE(model$simulation_params$store_reaction_firings),
  store_reaction_propensities = isTRUE(model$simulation_params$store_reaction_propensities),
  compute_cellwise_grn = isTRUE(model$simulation_params$compute_cellwise_grn),
  distance_metric = model$distance_metric,
  generate_experiment_seconds = generate_experiment_seconds
)
write_json(settings_core, file.path(output_dir, "settings.json"), auto_unbox = TRUE, pretty = TRUE)
settings_text <- c(
  "# Full dyngen settings and provenance",
  paste0("# generated_utc: ", format(Sys.time(), tz = "UTC", usetz = TRUE)),
  capture.output(print(settings_core)), "", "## numbers", capture.output(print(model$numbers)),
  "## tf_network_params", capture.output(print(model$tf_network_params)),
  "## feature_network_params", capture.output(print(model$feature_network_params)),
  "## kinetics_params", capture.output(print(model$kinetics_params)),
  "## gold_standard_params", capture.output(print(model$gold_standard_params)),
  "## simulation_params", capture.output(print(model$simulation_params)),
  "## experiment_params", capture.output(print(model$experiment_params)),
  "## sessionInfo", capture.output(sessionInfo())
)
writeLines(settings_text, file.path(output_dir, "settings.txt"))
saveRDS(model, file.path(output_dir, "model.rds"), compress = "xz")
checkpoint <- list(
  status = "complete", task = "orient/S21b", source_key = args$key,
  key = settings_core$key, backbone = args$backbone, seed = args$seed,
  timepoints = args$timepoints, cells = nrow(dataset$counts_spliced),
  genes = ncol(dataset$counts_spliced),
  fork_cells = if (args$backbone == "bifurcating") sum(
    (obs$truth_backbone_from == "sB" & obs$truth_backbone_to == "sBmid") |
      (obs$truth_backbone_from == "sBmid" &
         obs$truth_backbone_to %in% c("sC", "sD") & obs$truth_backbone_percentage < .15)
  ) else NA_integer_,
  generate_experiment_seconds = generate_experiment_seconds,
  generated_utc = format(Sys.time(), tz = "UTC", usetz = TRUE)
)
write_json(checkpoint, file.path(output_dir, "checkpoint.json"), auto_unbox = TRUE, pretty = TRUE)

data_dir <- file.path(root, "benchmarks/dyngen/data")
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)
h5ad_path <- file.path(data_dir, paste0(settings_core$key, ".h5ad"))
python <- file.path(root, ".sandbox/bin/python")
export_status <- system2(
  python,
  args = c("scripts/s19_export.py", "--raw", output_dir, "--output", h5ad_path,
           "--key", settings_core$key, "--backbone", args$backbone,
           "--seed", as.character(args$seed))
)
if (!is.null(export_status) && export_status != 0L) stop("s19_export.py failed with status ", export_status)

export_checkpoint_path <- file.path(root, "benchmarks/dyngen/checkpoints",
                                    paste0(settings_core$key, ".json"))
export_checkpoint <- fromJSON(export_checkpoint_path, simplifyVector = FALSE)
export_checkpoint$task <- "orient/S21b"
export_checkpoint$source_key <- args$key
export_checkpoint$timepoints <- args$timepoints
export_checkpoint$fork_cells <- checkpoint$fork_cells
write_json(export_checkpoint, export_checkpoint_path, auto_unbox = TRUE, pretty = TRUE)
checkpoint$h5ad_path <- file.path("benchmarks/dyngen/data", paste0(settings_core$key, ".h5ad"))
checkpoint$h5ad_sha256 <- export_checkpoint$sha256
checkpoint$h5ad_bytes <- export_checkpoint$bytes
write_json(checkpoint, file.path(root, "benchmarks/dyngen/s21b/checkpoints",
                                 paste0(settings_core$key, ".json")),
           auto_unbox = TRUE, pretty = TRUE)
cat(toJSON(checkpoint, auto_unbox = TRUE, pretty = TRUE), "\n")
