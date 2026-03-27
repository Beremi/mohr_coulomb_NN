# Repository Entry Point for Agents and Maintainers

This guide extends [README.md](../README.md). The README stays the contract-level overview and quick-start. This file is the clearer map for future agents and maintainers who need to answer two questions quickly:

1. what lives where in the repo
2. what else must be updated when something changes

## What This Repository Is For

This repository builds a local neural surrogate for the 3D Mohr-Coulomb constitutive update used in slope-stability finite elements.

The core loop is:

1. evaluate the exact constitutive update
2. generate synthetic or export-derived datasets
3. train a PyTorch surrogate
4. evaluate it on fixed holdouts
5. turn experiment artifacts into linked reports

The target is a pointwise constitutive map, not a global PDE operator. That is the reason the repo is organized around constitutive structure, branch geometry, dataset schema, and experiment provenance rather than full-solver orchestration.

## Start Here

Read these in order when you are new to the repo:

1. [README.md](../README.md)
2. [progress.md](../progress.md)
3. [repository_general_report_20260326.md](repository_general_report_20260326.md)
4. [repository_general_report_metric_dictionary_20260326.md](repository_general_report_metric_dictionary_20260326.md)
5. [cover_layer_comprehensive_report_20260326.md](cover_layer_comprehensive_report_20260326.md)
6. [repository_general_report_appendix_20260326.md](repository_general_report_appendix_20260326.md)
7. [mohr_coulomb.md](mohr_coulomb.md)
8. [dataset_creation.md](dataset_creation.md)
9. [full_export_inspection.md](full_export_inspection.md)
10. [report.md](../report.md)

If you only need the shortest orientation path, read:

- [progress.md](../progress.md)
- [repository_general_report_20260326.md](repository_general_report_20260326.md)
- [repository_general_report_metric_dictionary_20260326.md](repository_general_report_metric_dictionary_20260326.md)
- [cover_layer_comprehensive_report_20260326.md](cover_layer_comprehensive_report_20260326.md)
- [repository_general_report_appendix_20260326.md](repository_general_report_appendix_20260326.md)

## Repo Map

### Stable library code

- [`src/mc_surrogate/mohr_coulomb.py`](../src/mc_surrogate/mohr_coulomb.py): exact 3D Mohr-Coulomb constitutive update and branch logic
- [`src/mc_surrogate/materials.py`](../src/mc_surrogate/materials.py): elastic moduli and Davis reduction
- [`src/mc_surrogate/branch_geometry.py`](../src/mc_surrogate/branch_geometry.py): branch-surface geometry helpers
- [`src/mc_surrogate/sampling.py`](../src/mc_surrogate/sampling.py): synthetic dataset generation
- [`src/mc_surrogate/data.py`](../src/mc_surrogate/data.py): HDF5 schema and I/O
- [`src/mc_surrogate/models.py`](../src/mc_surrogate/models.py): feature builders and model factories
- [`src/mc_surrogate/training.py`](../src/mc_surrogate/training.py): training, validation, checkpointing, and metrics
- [`src/mc_surrogate/inference.py`](../src/mc_surrogate/inference.py): inference wrapper for checkpoints
- [`src/mc_surrogate/real_export.py`](../src/mc_surrogate/real_export.py): sampled real-export loading
- [`src/mc_surrogate/full_export.py`](../src/mc_surrogate/full_export.py): full FE export inspection and cover-layer extraction
- [`src/mc_surrogate/real_materials.py`](../src/mc_surrogate/real_materials.py): material-family inference for real exports

### Thin CLI scripts

- [`scripts/generate_dataset.py`](../scripts/generate_dataset.py): default synthetic data generation
- [`scripts/train_surrogate.py`](../scripts/train_surrogate.py): baseline training entry point
- [`scripts/evaluate_model.py`](../scripts/evaluate_model.py): evaluation entry point
- [`scripts/inspect_dataset.py`](../scripts/inspect_dataset.py): quick HDF5 inspection
- [`scripts/plot_training_history.py`](../scripts/plot_training_history.py): training curves
- [`scripts/plot_path_comparison.py`](../scripts/plot_path_comparison.py): path comparison plots
- [`scripts/compare_with_octave.py`](../scripts/compare_with_octave.py): cross-check against the Octave reference

### Experiment campaign layer

- [`scripts/experiments/`](../scripts/experiments): long-running studies, sweep drivers, report builders, post-training loops, and export preprocessing
- [`experiment_runs/`](../experiment_runs): canonical artifact store for datasets, checkpoints, summaries, plots, histories, and logs
- [`notebook_runs/`](../notebook_runs): lighter notebook-derived outputs and demos

### Reports and theory

- [`docs/`](.): theory notes, experiment logs, retained prehistory, and the canonical closeout set
- [`docs/repository_general_report_20260326.md`](repository_general_report_20260326.md): current repository-wide decision report
- [`docs/repository_general_report_metric_dictionary_20260326.md`](repository_general_report_metric_dictionary_20260326.md): canonical metric glossary and alias map
- [`docs/cover_layer_comprehensive_report_20260326.md`](cover_layer_comprehensive_report_20260326.md): dedicated closeout for the local cover-layer line and its generator / routing prehistory
- [`docs/repository_general_report_appendix_20260326.md`](repository_general_report_appendix_20260326.md): retained-docs catalog for the current `docs/**/*.md` library
- [`docs/tasks/projection_student_work_packet_20260326.md`](tasks/projection_student_work_packet_20260326.md): normalized task memo for the live projection-student packet
- [`docs/executions/projection_student_work_packet_20260326.md`](executions/projection_student_work_packet_20260326.md): standalone execution report for the same packet
- [`report.md`](../report.md): literature review and state-of-the-art framing retained as background

### Tests

- [`tests/test_constitutive.py`](../tests/test_constitutive.py): branch behavior and constitutive correctness
- [`tests/test_sampling.py`](../tests/test_sampling.py): dataset generation coverage
- [`tests/test_training.py`](../tests/test_training.py): training and checkpoint smoke coverage
- [`tests/test_full_export.py`](../tests/test_full_export.py): export and FE-trace handling
- [`tests/test_fe_p2_tetra.py`](../tests/test_fe_p2_tetra.py): tetrahedral FE helpers

## Main Workflows

### 1. Synthetic pipeline

Use this when the question is about the exact constitutive map in isolation.

1. Generate an HDF5 dataset with [`scripts/generate_dataset.py`](../scripts/generate_dataset.py).
2. Train a surrogate with [`scripts/train_surrogate.py`](../scripts/train_surrogate.py).
3. Evaluate it with [`scripts/evaluate_model.py`](../scripts/evaluate_model.py).
4. Plot training curves with [`scripts/plot_training_history.py`](../scripts/plot_training_history.py).

Important definitions live in:

- [`src/mc_surrogate/data.py`](../src/mc_surrogate/data.py)
- [`src/mc_surrogate/models.py`](../src/mc_surrogate/models.py)
- [`README.md`](../README.md)

### 2. Real-export pipeline

Use this when the question is about real solver structure, transfer, or FE-compatible states.

1. Start with [`full_export_inspection.md`](full_export_inspection.md).
2. Inspect or convert exports with [`src/mc_surrogate/full_export.py`](../src/mc_surrogate/full_export.py) and [`src/mc_surrogate/real_export.py`](../src/mc_surrogate/real_export.py).
3. Build sampled datasets with [`scripts/experiments/preprocess_real_sim_export.py`](../scripts/experiments/preprocess_real_sim_export.py).
4. Train real-data studies with [`scripts/experiments/staged_real_sim_training.py`](../scripts/experiments/staged_real_sim_training.py).
5. Validate with [`scripts/experiments/run_real_sim_validation.py`](../scripts/experiments/run_real_sim_validation.py).

### 3. Experiment-to-report pipeline

Use this when the question is about reproducibility and repo history.

1. run or rerun a campaign from [`scripts/experiments/`](../scripts/experiments)
2. store artifacts under [`experiment_runs/`](../experiment_runs)
3. generate or update a Markdown report under [`docs/`](.)
4. link the report back to the exact artifact directory and producing script

The docs are not just prose. Many of them are reproducible views over run artifacts.

## How To Expand The Repo Safely

This is the maintenance manual to follow whenever something changes.

### If you change the constitutive law

Update all of:

- [`src/mc_surrogate/mohr_coulomb.py`](../src/mc_surrogate/mohr_coulomb.py)
- [`src/mc_surrogate/materials.py`](../src/mc_surrogate/materials.py) if reduction or material inputs changed
- [`src/mc_surrogate/branch_geometry.py`](../src/mc_surrogate/branch_geometry.py) if decision surfaces changed
- [`docs/mohr_coulomb.md`](mohr_coulomb.md)
- [`tests/test_constitutive.py`](../tests/test_constitutive.py)

### If you change dataset schema or feature inputs

Update all of:

- [`src/mc_surrogate/data.py`](../src/mc_surrogate/data.py)
- [`src/mc_surrogate/sampling.py`](../src/mc_surrogate/sampling.py)
- [`src/mc_surrogate/models.py`](../src/mc_surrogate/models.py) if features changed
- [`scripts/generate_dataset.py`](../scripts/generate_dataset.py)
- [`scripts/inspect_dataset.py`](../scripts/inspect_dataset.py) when inspection output needs to stay truthful
- [`dataset_creation.md`](dataset_creation.md)
- [`tests/test_sampling.py`](../tests/test_sampling.py)

### If you change model kinds, training behavior, or evaluation

Update all of:

- [`src/mc_surrogate/models.py`](../src/mc_surrogate/models.py)
- [`src/mc_surrogate/training.py`](../src/mc_surrogate/training.py)
- [`src/mc_surrogate/inference.py`](../src/mc_surrogate/inference.py) if deployment behavior changes
- [`scripts/train_surrogate.py`](../scripts/train_surrogate.py)
- [`scripts/evaluate_model.py`](../scripts/evaluate_model.py)
- [`tests/test_training.py`](../tests/test_training.py)
- the relevant report in [`docs/`](.) if the repo’s recommended baseline changes

### If you change real-export handling

Update all of:

- [`src/mc_surrogate/real_export.py`](../src/mc_surrogate/real_export.py)
- [`src/mc_surrogate/full_export.py`](../src/mc_surrogate/full_export.py)
- [`src/mc_surrogate/real_materials.py`](../src/mc_surrogate/real_materials.py) if material-family interpretation changed
- [`scripts/experiments/preprocess_real_sim_export.py`](../scripts/experiments/preprocess_real_sim_export.py)
- [`full_export_inspection.md`](full_export_inspection.md) if assumptions or schema notes changed
- [`tests/test_full_export.py`](../tests/test_full_export.py)

### If you add a new experiment campaign

Do all of the following:

1. add the runner under [`scripts/experiments/`](../scripts/experiments)
2. choose a stable output directory under [`experiment_runs/`](../experiment_runs)
3. save machine-readable artifacts such as `summary.json`, `history.json`, `.csv`, and plots where applicable
4. add or update a Markdown report in [`docs/`](.)
5. link the report to the exact artifact directory and, if helpful, the producing script

### If you add a new artifact type

Document it before treating it as canonical:

- owning script
- filename convention
- expected consumer
- whether it is raw data, derived dataset, checkpoint, benchmark, or report artifact

The right place is usually the nearest relevant doc in [`docs/`](.) and sometimes also [README.md](../README.md) if the workflow contract changed.

### If you change the repo’s public story

If the intended workflow, recommended model family, or key terminology changed, update:

- [README.md](../README.md)
- this file
- the most relevant strategy doc in [`docs/`](.)

Do not let the README and docs drift apart.

## Quick Verification Checklist

After substantive changes, run the most relevant focused tests first:

- [`tests/test_constitutive.py`](../tests/test_constitutive.py)
- [`tests/test_sampling.py`](../tests/test_sampling.py)
- [`tests/test_training.py`](../tests/test_training.py)
- [`tests/test_full_export.py`](../tests/test_full_export.py)

Then confirm that the affected report or doc still points to the correct artifact paths.

## Rule Of Thumb

Use [`src/mc_surrogate/`](../src/mc_surrogate) when you need stable reusable behavior, use [`scripts/experiments/`](../scripts/experiments) when you need a reproducible campaign, and use [`docs/`](.) when you need the rationale and historical context behind a campaign.
