# Repository General Report Metric Dictionary 20260326

## Canonical Usage

- Raw artifact field names are the canonical source of truth.
- For packets2-4 and Program X, the primary decision metrics were `broad_plastic_mae`, `hard_plastic_mae`, `hard_p95_principal`, and `yield_violation_p95`.
- `hard_rel_p95_principal` was a supporting severity metric, not a top-level stop/continue gate.
- All stress-like values use repository stress units. Yield-violation metrics are dimensionless residual-style diagnostics and should be as close to zero as possible.

## Core Metrics

| Canonical term | Raw key | What it measures | Better | Typical aliases in prose | Decision role |
| --- | --- | --- | --- | --- | --- |
| Broad MAE | `broad_mae` | All-row stress MAE on the broad panel, including exact elastic rows | Lower | broad stress MAE, broad all-row MAE | Secondary context |
| Broad plastic MAE | `broad_plastic_mae` | Stress MAE on plastic rows in the broad panel | Lower | broad plastic stress MAE | Primary |
| Hard MAE | `hard_mae` | All-row stress MAE on the hard panel | Lower | hard stress MAE, hard all-row MAE | Secondary context |
| Hard plastic MAE | `hard_plastic_mae` | Stress MAE on plastic rows in the hard panel | Lower | hard plastic stress MAE | Primary |
| Hard p95 principal | `hard_p95_principal` | 95th percentile absolute principal-stress error on the hard panel | Lower | hard p95 max-principal abs error, hard principal p95 | Primary |
| Hard relative p95 principal | `hard_rel_p95_principal` | 95th percentile repository-relative principal-stress error on the hard panel | Lower | hard repo-relative principal p95 | Supporting |
| Yield violation p95 | `yield_violation_p95` | 95th percentile admissibility violation after decode | Lower | yield p95, yield tail p95 | Primary as a stability gate |
| Yield violation max | `yield_violation_max` | Worst admissibility violation after decode | Lower | yield max | Supporting tail diagnostic |

## Evaluation-Mode Terms

| Canonical term | Meaning | Better | Notes |
| --- | --- | --- | --- |
| Predicted evaluation | Actual deployed inference path for the packet | Lower metrics | For packet3 this means learned hard routing; for packet4 it means deployed soft routing |
| Oracle evaluation | Evaluation with non-deployed information that the final model would not have at inference time | Lower metrics | Usually teacher-forced true-branch or oracle-route evaluation |
| Deployed winner | The selected model/checkpoint/temperature under the packet’s real deployment rule | Lower metrics | This is the number that matters for go/no-go decisions |
| Oracle winner | The best or reported oracle-mode number for the same representation family | Lower metrics | Ceiling diagnostic only; not a deployment result |

## Exactness Audit Metrics

| Canonical term | Raw key | What it measures | Better |
| --- | --- | --- | --- |
| Round-trip mean abs | `round_trip_mean_abs` or `reconstruction.mean_abs` | Mean absolute difference between exact target and representation round-trip decode | Lower |
| Round-trip max abs | `round_trip_max_abs` or `reconstruction.max_abs` | Maximum absolute round-trip difference | Lower |
| Branch agreement | `branch_agreement` | Fraction of rows whose audited branch label matches the source branch label | Higher |
| Stage 0 pass | `stage0_pass` | Whether the required exactness checks passed | `True` |
| Exactness yield p95 / max | `yield.p95`, `yield.max` | Admissibility tail after an exactness round-trip audit | Lower |

## Panel and Scope Conventions

- `broad_*` panels refer to the full grouped validation/test populations.
- `hard_*` panels refer to geometry- and tail-focused subsets defined in the panel sidecar.
- `plastic` metrics remove exact elastic rows from the metric pool.
- All-row metrics can look better than plastic-only metrics because exact elastic rows are solved analytically and contribute zero error by construction.

## Primary Decision Bars Used in the Repo

### Credibility anchor for full-coverage replacement

- `broad_plastic_mae <= 5.771003`
- `hard_plastic_mae <= 6.949571`
- `hard_p95_principal <= 76.398964`
- `yield_violation_p95 <= 0.100133`

### Internal `DS` bar used by the surface packets

- `broad_plastic_mae <= 15`
- `hard_plastic_mae <= 18`
- `hard_p95_principal <= 150`
- `yield_violation_p95 <= 1e-6`

### Packet4 continue bar

- `broad_plastic_mae <= 25`
- `hard_plastic_mae <= 28`
- `hard_p95_principal <= 250`
- `yield_violation_p95 <= 1e-6`

### Program X oracle bars

- Minimum oracle success:
  - `broad_plastic_mae <= 25`
  - `hard_plastic_mae <= 28`
  - `hard_p95_principal <= 250`
  - `yield_violation_p95 <= 1e-6`
- Strong oracle success:
  - `broad_plastic_mae <= 20`
  - `hard_plastic_mae <= 24`
  - `hard_p95_principal <= 200`
  - `yield_violation_p95 <= 1e-6`

## Normalization Notes Applied in the Consolidated Report

- Packet2 validation/test metrics are taken from the packet2 checkpoint-selection and winner-test JSON artifacts.
- Packet3 predicted and oracle validation metrics are taken from `experiment_runs/real_sim/nn_replacement_surface_20260324_packet3/exp1_branch_structured_surface/architecture_summary_val.csv`.
- Packet3 oracle validation `yield_violation_p95` is normalized to the raw CSV value `5.414859e-08`; prose summaries often repeated the predicted-route yield value instead.
- Packet3 oracle held-out test metrics are intentionally left unavailable because no canonical raw oracle-test artifact was located.
- Packet4 deployed/oracle validation metrics are taken from packet4 selection and decision JSONs; packet4 test metrics are taken from the winner-test JSON.
- Program X benchmark, bars, and reference baselines are taken from `baseline_manifest.json` and `reference_metrics.json`; Stage 0 exactness comes from `stage0_exactness_summary.json`; overall oracle validation/test metrics come from `oracle_val_summary.json` and `oracle_test_summary.json`.
