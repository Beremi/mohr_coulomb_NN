# Repository General Report 20260326

## Executive Summary

- The current controlling state is no longer the Program X terminal stop. The March 26 strategic override reopened the direct local NN replacement line only along the `projection-student` route. The current repository state is `continue_projection_student`, while `DS` remains blocked.
- The best historical deployed packet-family direct-replacement model remains packet2: exact elastic dispatch plus a plastic-only admissible `g, rho` surface model. It remained the best deployed validation/test winner among packets1-4, but it is no longer the best admissibility-preserving result in the repo after the projection-student reopen.
- Earlier baselines outside the March 24 packet benchmark matter for context but not for the final ranking: synthetic-only studies reached sub-1 MAE in-domain, staged mixed-material real-data training reached `4.861 / 4.754` MAE on its own fixed splits, and cover-layer-only fitted-refresh or routed-expert studies reached `1.495` single-model MAE and `2.808` routed MAE on local holdouts. Those results used different scopes or had local / holdout-informed caveats, so they were never the controlling packet benchmark. The full local cover-layer prehistory is now consolidated in `docs/cover_layer_comprehensive_report_20260326.md`.
- Packet3 proved that branch-local signal exists because oracle routing beat packet2 on validation, but predicted routing failed in deployment and oracle performance still missed the internal bars.
- Packet4 was the final bounded surface-family test. It removed hard argmax routing, kept admissibility stable, but still lost to packet2 on the core deployed validation stress metrics and did not clear the continue bar.
- Program X was the final representation-change test in the pre-reopen line. Stage 0 exactness proved the exact-latent dataset and decode were numerically correct, but the teacher-forced branchwise oracle was catastrophically worse than packet2, packet3-oracle, and packet4. That result closed the old direct-replacement family.
- The projection-student reopen changed the state because Phase 0 succeeded decisively: exact ordered-principal projection of the raw teacher drove validation/test yield tails from about `1e-1` to about `2e-8` while keeping stress quality near the raw-teacher band. The bounded Phase 2 projected student then beat packet2, packet3 oracle, and packet4 on the three core validation stress metrics while preserving numerical-zero yield, but it still missed the stronger `DS` bar.

## Program Timeline / Program Map

| Family | Main absorbed sources | Purpose | Representation and benchmark | Key outcome | Terminal decision |
| --- | --- | --- | --- | --- | --- |
| Synthetic demo / high-capacity foundations | `docs/demo_training_experiments.md`, `docs/high_capacity_study.md` | Prove the exact constitutive map is learnable on synthetic data and establish in-domain training baselines | Synthetic datasets, principal/raw surrogates, notebook-scale then higher-capacity repeated sweeps | In-domain fit became strong and repeatable, but these results did not transfer directly to deployment-shaped real solver states | Keep only as learnability context; synthetic-only ranking is not a deployment decision rule |
| Mixed-material real-sim transition | `docs/real_sim_validation.md`, `docs/real_sim_retraining.md`, `docs/real_sim_long_sweep.md`, `docs/staged_real_sim_training.md`, `docs/relative_error_visual_report.md` | Move from synthetic-only evaluation to real sampled constitutive states exported from the slope-stability solve | Real sampled mixed-material datasets, first on `256`-per-call then broader `512`-per-call splits | Synthetic-only baselines collapsed on real states, real-data retraining rescued the line, and staged training produced the strongest broad historical real-data baseline before the packet programs | Use as historical baseline only; later programs changed the target representation and benchmark |
| Per-material split / hardcase line | `docs/per_material_*.md` | Test whether material specialization solves the mixed-material difficulty | One model per material family, then hybrid hardcase augmentation for the difficult families | Easier foundation families improved strongly, but the mixed-best aggregate still lost badly to the best mixed-material real-data baseline | Do not treat material splitting alone as the fix |
| Cover-layer branch-predictor prehistory | `report.md`, `docs/expert_briefing_20260323.md`, `docs/cover_layer_comprehensive_report_20260326.md` | Learn constitutive branch structure and diagnose why synthetic transfer failed before a full solver-facing surrogate existed | Branch predictors, strain-only variants, element-state variants, synthetic and local real panels | Branch-local signal existed, but predicted branch use remained too brittle for deployment; this work supplied geometry insight and failure diagnosis rather than a deployable constitutive replacement | Consolidated into a dedicated local closeout; not a repository-wide control line |
| Cover-layer exact-domain / fitted-refresh local line | `docs/cover_layer_comprehensive_report_20260326.md` | Test whether exact-domain cover-layer training or generator correction can produce a strong local surrogate on a fixed material family | Cover-layer-only exact-domain datasets, fitted-refresh synthetic generators, local real holdouts | Exact-domain training became credible and fitted-refresh produced the strongest single-model local cover-layer baseline in the repo | Valuable local prehistory, but not a repository-wide control benchmark |
| Cover-layer routed shadow-candidate line | `docs/cover_layer_comprehensive_report_20260326.md` | Push the strongest local cover-layer route toward solver-facing tail safety with gating and edge experts | Cover-layer-only routed ensemble, hard-mined edge experts, holdout-informed tail-safety phase | A local routed candidate became strong enough for a solver shadow test, but it stayed material-local and relied on holdout-informed mining | Never promoted to general control; treated as historical local candidate only |
| Hybrid safe pivot | `report3.md`, `docs/hybrid_pivot_*.md` | Keep exact constitutive authority near hard regions and learn only a plastic interior corrector | Exact elastic dispatch + exact fallback + learned residual, March 23 grouped-call split (`seed=20260323`) | Wrapper-only runs could lower accepted-region error, but learned coverage was too small; the learned path was not solver-usable | No safe-hybrid promotion |
| Hybrid gate redesign | `report4.md`, `docs/hybrid_gate_redesign_*.md`, `docs/expert_complete_report_20260324.md` | Recover a deployable learned region by redesigning the risk/coverage gate on the March 24 benchmark | Candidate-B gate redesign on the March 24 validation-first split (`seed=20260324`) | Candidate B could look strong on accepted-region stress but still failed admissibility badly; gate redesign did not recover a deployable frontier | Close the redesign line |
| Exact-authority return-mapping acceleration scout | `report5.md`, `docs/return_mapping_accel_*.md`, `docs/expert_complete_report_20260324.md` | Use ML only to help the exact constitutive operator instead of owning the final stress | Exact closed-form constitutive update plus profiling/branch-hint shortlist proxies on the March 24 benchmark | Exactness was preserved, but the in-repo shortlist proxy slowed the operator instead of speeding it up | Do not continue in-repo exact-authority acceleration |
| Packet1: admissible-coordinate opening packet | `report6.md`, `docs/nn_replacement_abr_*.md` | Make admissibility structural by predicting exact principal-space `a, b, r` coordinates | Full-coverage branchless admissible-coordinate network on the March 24 benchmark | Structural exactness passed and yield stayed at numerical zero, but stress accuracy was a clear no-go | Move to plastic-only surface coordinates |
| Packet2: branchless plastic-surface winner | `report7.md`, `docs/nn_replacement_surface_exp0_20260324.md`, `docs/nn_replacement_surface_exp1_branchless_20260324.md`, `docs/nn_replacement_surface_readable_summary_20260324.md`, `docs/nn_replacement_surface_execution_20260324.md` | Keep elastic handling exact, learn only plastic rows, and use a better-conditioned admissible surface target | Exact elastic dispatch plus plastic-only `g, rho`, March 24 validation-first split | Material improvement over packet1 and the best deployed direct-replacement result in the repo | Continue once, but only with a bounded redesign |
| Packet3: branch-structured surface | `report8.md`, `docs/nn_replacement_surface_packet3_*.md`, `docs/expert_current_state_report_20260324.md` | Test whether explicit branch structure beats packet2 | Hard-routed branch-specialized admissible surface on the March 24 benchmark | Oracle routing beat packet2 on validation, but predicted routing lost to packet2 and oracle still missed the internal bar | Routing identified as an immediate blocker; at most one more bounded packet justified |
| Packet4: soft admissible atlas | `report9.md`, `report10.md`, `docs/nn_replacement_surface_packet4_*.md`, `experiment_runs/real_sim/nn_replacement_surface_20260324_packet4/exp1_soft_atlas_surface/packet4_vs_packet2_packet3_val.md` | Remove packet3 argmax routing while keeping admissibility exact and branch-local charts | Soft atlas over `smooth / left / right / apex`, March 24 benchmark, post-hoc validation temperature sweep | Deployed winner remained worse than packet2 on the core stress metrics; packet4 oracle diagnostics were worse than deployed routing | Stop the direct surface-family line |
| Program X exact-latent family | `report11.md`, `docs/nn_replacement_exact_latent_*.md`, `experiment_runs/real_sim/nn_replacement_exact_latent_20260325/exp0_baseline_freeze/execution_plan.md`, `experiment_runs/real_sim/nn_replacement_exact_latent_20260325/exp4_oracle_eval/programx_vs_packet2_packet3_packet4_val.md`, `progress.md` | Test whether exact branchwise latent variables rescue direct replacement if routing is removed from the question | Teacher-forced branchwise exact-latent oracle on the March 24 benchmark | Stage 0 exactness passed, but the oracle was catastrophically worse than all recent direct baselines and failed every meaningful bar | Retire the direct local NN replacement line; no routing follow-on; `DS` remains blocked |
| Projection-student reopen route | `tasks1.md`, `progress.md`, `docs/tasks/projection_student_work_packet_20260326.md`, `docs/executions/projection_student_work_packet_20260326.md`, `experiment_runs/real_sim/projection_student_20260326/exp0_phase0_audit/phase0_summary.json`, `experiment_runs/real_sim/projection_student_20260326/exp2_projected_student/phase2_summary.json` | Separate stress prediction from admissibility by projecting ordered principal stress onto the exact admissible MC set, then train a bounded projected student around that layer | Frozen March 24 raw teacher plus exact ordered-principal projection audit, then bounded projected-student sweep on the same benchmark | Phase 0 preserved raw-teacher stress quality while driving yield to numerical zero; the bounded medium projected student beat packet2, packet3 oracle, and packet4 on validation while remaining admissible, but stayed far from raw-teacher stress quality and missed the `DS` reopen bar | Continue only the projection-student line; keep `DS` blocked |

### Historical Baselines Before the March 24 Packet Benchmark

These older baselines are important for context, but they are not directly comparable to the packet2/3/4 and Program X tables because they used different scopes, splits, or problem definitions.

| Family | Canonical raw sources | Best scope-specific result | Why it mattered | Why it did not become the final control |
| --- | --- | --- | --- | --- |
| Synthetic demo and high-capacity studies | `experiment_runs/demo_sweep/results.csv`; `experiment_runs/high_capacity/study_pilot/aggregate_summary.json` | Demo best `focused_capped_large`: stress MAE `0.6948`, branch acc `0.998`; repeated high-capacity synthetic study: mean test stress MAE `0.9961`, branch acc about `0.997` | Proved the exact map was learnable and that data-scale control mattered more than blindly increasing width/depth | The benchmark was synthetic-only and later real-sim studies exposed a severe deployment distribution shift |
| Mixed-material real-sim baseline | `experiment_runs/real_sim/baseline_validation/synthetic_test_metrics.json`; `experiment_runs/real_sim/baseline_validation/real_metrics.json`; `experiment_runs/real_sim/staged_20260312/staged_results_manual.json` | Synthetic baseline transferred catastrophically (`75.027` real MAE, branch acc `0.434`), then staged real-data `512x6` reached `4.861` primary / `4.754` cross MAE with branch acc `0.942 / 0.943` | Established that real sampled data was mandatory and produced the strongest broad historical real-data baseline before the packet line | It remained a broad regression baseline without the exact admissibility structure later packet work targeted |
| Per-material split and hybrid hardcase studies | `experiment_runs/real_sim/per_material_synth_to_real_20260312/summary.json`; `experiment_runs/real_sim/per_material_hybrid_hardcases_20260312/summary.json` | Mixed-best aggregate improved to `10.305` primary / `10.433` cross MAE, but still lost badly to the staged mixed-material baseline `4.861 / 4.754` | Showed that material specialization helped easy families but did not solve the hard plastic families | Material splitting alone was not a sufficient answer |
| Cover-layer exact-domain and fitted-refresh local line | `experiment_runs/real_sim/cover_layer_cyclic_20260312/sweep_summary.json`; `experiment_runs/real_sim/cover_layer_fitted_refresh_heavy_w1024_20260312/cover_raw_branch_w1024_d6_local_noise/summary.json` | Exact-domain cyclic best reached about `4.712` primary MAE; later fitted-refresh `w1024 d6 local_noise` reached `1.495` real-test MAE with branch acc `0.958` | Produced the strongest single-model local cover-layer baseline and proved that generator correction could matter more than pure capacity | The scope was one fixed material family on local holdouts, not the later repository-wide March 24 packet benchmark |
| Cover-layer routed shadow candidate | `experiment_runs/real_sim/cover_layer_tail_safety_20260313/routes_summary.json` | `edge_hard_mined_gate_raw_threshold_t0.65` reached real MAE `2.808`, RMSE `16.020`, p99 relative error `1.539`, branch acc `0.956` | Demonstrated the strongest local routed candidate and earned a solver shadow-test go decision | It relied on holdout-informed hard mining and remained cover-layer-only, so it was never promoted to the repository-wide control line |

## Benchmark and Split Definition

The final comparison benchmark for all recent decision packets is the March 24 validation-first grouped-call split frozen under Program X and then reused unchanged by the projection-student reopen.

### Benchmark History

| Benchmark | Source / grouped dataset | Split seed | Split unit | Split fractions | Samples per call | Used by |
| --- | --- | ---: | --- | --- | ---: | --- |
| March 23 pivot benchmark | `constitutive_problem_3D_full.h5` -> `experiment_runs/real_sim/hybrid_pivot_20260323/real_grouped_sampled_512.h5` | `20260323` | constitutive call | `70 / 15 / 15` | `512` | Hybrid safe pivot only |
| March 24 canonical benchmark | `constitutive_problem_3D_full.h5` -> `experiment_runs/real_sim/hybrid_gate_redesign_20260324/real_grouped_sampled_512.h5` | `20260324` | constitutive call | `70 / 15 / 15` | `512` | Hybrid redesign closeout, return-mapping scout, packets1-4, Program X, projection-student reopen |

### Canonical March 24 Benchmark

- Source H5: `constitutive_problem_3D_full.h5`
- Grouped dataset: `experiment_runs/real_sim/hybrid_gate_redesign_20260324/real_grouped_sampled_512.h5`
- Panel sidecar: `experiment_runs/real_sim/hybrid_gate_redesign_20260324/panels/panel_sidecar.h5`
- Split seed: `20260324`
- Split unit: constitutive call
- Split fractions: `70 / 15 / 15`
- Samples per call: `512`

| Split / panel | Rows |
| --- | ---: |
| `train` | 198656 |
| `val` | 42496 |
| `test` | 42496 |
| `broad_val` | 42496 |
| `broad_test` | 42496 |
| `hard_val` | 21198 |
| `hard_test` | 22705 |
| `ds_valid` | 73261 |
| `ds_valid_val` | 11224 |
| `ds_valid_test` | 9839 |

Panel conventions used by the recent packets:

- `broad_*` panels are the full grouped validation/test populations.
- `hard_*` panels are geometry- and tail-focused subsets built from the panel sidecar.
- `ds_valid*` panels are the subset used as the tangent-readiness gate; they remain blocked because no learned forward model has yet met the stronger projection-student `DS` reopening bar.

## Metric Dictionary Summary with Aliases

The companion file `docs/repository_general_report_metric_dictionary_20260326.md` is the full canonical dictionary. The condensed summary below captures the terms used in the recent packets.

| Canonical term | Raw key | Meaning | Better | Decision role |
| --- | --- | --- | --- | --- |
| Broad MAE | `broad_mae` | All-row stress MAE on the broad panel, including exact elastic rows | Lower | Secondary context |
| Broad plastic MAE | `broad_plastic_mae` | Stress MAE on plastic rows in the broad panel | Lower | Primary |
| Hard MAE | `hard_mae` | All-row stress MAE on the hard panel | Lower | Secondary context |
| Hard plastic MAE | `hard_plastic_mae` | Stress MAE on plastic rows in the hard panel | Lower | Primary |
| Hard p95 principal | `hard_p95_principal` | 95th percentile absolute principal-stress error on the hard panel | Lower | Primary |
| Hard relative p95 principal | `hard_rel_p95_principal` | 95th percentile repository-relative principal-stress error on the hard panel | Lower | Supporting severity metric |
| Yield violation p95 / max | `yield_violation_p95`, `yield_violation_max` | Tail of admissibility violation after decode | Lower | Stability constraint; sometimes primary only as a gate |
| Predicted evaluation | varies by packet | Actual deployed routing / deployed winner evaluation | Lower | What matters for deployment |
| Oracle evaluation | varies by packet | Teacher-forced true-branch or oracle-route diagnostic evaluation | Lower | Ceiling diagnostic, not deployment |
| Exactness audit | `round_trip_*`, `branch_agreement`, yield audit fields | Whether the derived representation decodes back to the exact constitutive result | Lower for errors, higher for agreement | Required before trusting a new representation |

Normalization notes applied in this report:

- Raw artifact field names are canonical whenever prose summaries disagree.
- Packet3 oracle validation `yield_violation_p95` is normalized from the raw CSV value `5.414859e-08`; several prose docs repeated the predicted-route value instead.
- Packet3 oracle held-out test metrics were not normalized because no canonical raw oracle-test artifact was found. They are reported as unavailable rather than backfilled from prose.

## Normalized Result Tables

### Validation Comparison on the Canonical March 24 Benchmark

| Row | Mode | Winner / representation | Params | Broad plastic MAE | Hard plastic MAE | Hard p95 principal | Yield p95 | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Packet2 deployed winner | deployed | `p1_w128_d3`, branchless plastic-only `g, rho` | 35714 | 32.458569 | 37.824432 | 328.641602 | 6.173982e-08 | Best deployed direct winner |
| Packet3 predicted winner | predicted | `p3_w128_d3`, hard-routed branch-structured surface | 36488 | 38.164696 | 43.860016 | 399.468353 | 5.600788e-08 | Lost to packet2 in deployment |
| Packet3 oracle winner | oracle | `p3_w128_d3`, true-branch oracle route | 36488 | 29.512514 | 32.682789 | 302.360168 | 5.414859e-08 | Proved branch-local signal exists |
| Packet4 deployed winner | deployed | `film_w128_d3 @ T=1.25`, soft admissible atlas | 203784 | 41.599422 | 46.363712 | 374.894531 | 6.008882e-08 | Final surface-family packet; failed continue bar |
| Program X oracle | teacher-forced oracle | Exact-latent branchwise models plus analytic apex | 20737 / 9729 / 9729 | 250.392441 | 328.857300 | 2331.702637 | 1.258780e+00 | Catastrophic failure; closed the line |
| Projection-student Phase 0 | projected teacher | Raw mixed-material teacher plus exact MC projection | n/a | 5.877489 | 7.062693 | 76.398941 | 2.199150e-08 | Reopen audit passed decisively |
| Projection-student Phase 2 winner | projected student | `medium_exact`, bounded projected student | 29847 | 27.308128 | 29.599710 | 250.701462 | 2.270879e-08 | Beat packet2 and packet3 oracle on validation; `DS` bar still missed |

### Held-Out Test Comparison

| Row | Mode | Broad plastic MAE | Hard plastic MAE | Hard p95 principal | Hard rel p95 principal | Yield p95 | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Packet2 deployed winner | deployed | 33.603760 | 38.797607 | 331.668732 | 2.737226 | 5.991195e-08 | Best held-out deployed direct result |
| Packet3 predicted winner | predicted | 39.744358 | 45.307705 | 414.473572 | 3.309789 | 5.418011e-08 | Oracle test unavailable |
| Packet4 deployed winner | deployed | 43.238693 | 48.038338 | 388.482666 | 3.159766 | 5.825863e-08 | Stable yield, weak stress |
| Program X oracle | teacher-forced oracle | 285.859314 | 378.356659 | 2632.823242 | 27.600845 | 1.260843e+00 | Catastrophic held-out failure |
| Projection-student Phase 0 | projected teacher | 6.083902 | 7.344495 | 79.955505 | 0.612849 | 2.252151e-08 | Yield fixed with only mild stress drift from the raw teacher |
| Projection-student Phase 2 winner | projected student | 28.099649 | 30.332142 | 259.966797 | 1.879826 | 2.252200e-08 | Best held-out result of the reopen route, but still far from teacher quality |

### Projection-Student Reopen Summary

Raw teacher anchor versus projected teacher:

- validation raw `5.771003 / 6.949572 / 76.398941 / 1.001325e-01` became projected `5.877489 / 7.062693 / 76.398941 / 2.199150e-08`
- test raw `5.981674 / 7.241915 / 79.455521 / 1.061293e-01` became projected `6.083902 / 7.344495 / 79.955505 / 2.252151e-08`
- validation projection displacement `p50 / p95 / max`: `1.005166 / 11.713972 / 123.145325`
- test projection displacement `p50 / p95 / max`: `1.097234 / 12.012154 / 251.903427`

Bounded Phase 2 size-band sweep on validation:

| Run | Params | Broad plastic MAE | Hard plastic MAE | Hard p95 principal | Yield p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| small exact | 11767 | 29.981367 | 33.359127 | 295.993927 | 2.321422e-08 |
| medium exact | 29847 | 27.308128 | 29.599710 | 250.701462 | 2.270879e-08 |
| large exact | 41959 | 29.941349 | 31.859295 | 274.728699 | 2.344744e-08 |

Interpretation:

- Phase 0 proved that the direct-replacement line was not blocked by stress learnability alone; the exact projection layer could rescue admissibility without destroying a strong raw teacher
- Phase 2 proved that a bounded projected student could beat all previous admissible deployed or oracle-like direct baselines on validation
- Phase 2 also showed the remaining bottleneck clearly: even the winner stayed much worse than the projected teacher, so the learned student still did not reach the quality needed for a credible `DS` reopening

### Program X Stage 0 Exactness Audit

| Metric | Value |
| --- | ---: |
| Round-trip mean abs | 3.961551e-13 |
| Round-trip max abs | 2.328306e-10 |
| Yield violation p95 | 4.662968e-14 |
| Yield violation max | 1.060576e-11 |
| Branch agreement | 1.0 |
| Stage 0 pass | `True` |

### Program X Branchwise Validation Winners

| Branch | Winner | Params | Val stress-component MAE | Val principal p95 | Val yield p95 |
| --- | --- | ---: | ---: | ---: | ---: |
| `smooth` | `smooth_w96_d3` | 20737 | 275.604462 | 956.649841 | 1.278197 |
| `left_edge` | `left_edge_w64_d3` | 9729 | 274.984619 | 1876.733887 | 1.056348 |
| `right_edge` | `right_edge_w64_d3` | 9729 | 337.012878 | 3318.269531 | 1.320152 |
| `apex` | analytic only | 0 learned | n/a | n/a | n/a |

## Model-Family Comparison and Decision Logic

### Decision Bars

| Bar | Thresholds | Applied to | Outcome |
| --- | --- | --- | --- |
| Credibility anchor | `broad plastic <= 5.771003`, `hard plastic <= 6.949571`, `hard p95 <= 76.398964`, `yield p95 <= 0.100133` | March 24 full-coverage replacement credibility | No direct-replacement family came close |
| Internal `DS` bar | `<= 15 / <= 18 / <= 150 / <= 1e-6` | Packet2-4 direct replacement line | Failed by packets2-4 |
| Packet4 continue bar | `<= 25 / <= 28 / <= 250 / <= 1e-6` plus beat packet2 on the three core stress metrics | Final surface-family packet | Failed |
| Packet4 routing-gap bar | Cut packet3 oracle-vs-predicted gaps by at least half | Packet4 soft-atlas packet | Failed; oracle diagnostic was worse than deployed |
| Program X minimum oracle bar | `<= 25 / <= 28 / <= 250 / <= 1e-6` | Exact-latent teacher-forced oracle | Failed catastrophically |
| Program X strong oracle bar | `<= 20 / <= 24 / <= 200 / <= 1e-6` | Exact-latent teacher-forced oracle | Failed even more decisively |
| Projection-student continue bar | `<= 10 / <= 12 / <= 120 / <= 1e-6` | Phase 0 projection audit or Phase 2 projected student | Phase 0 passed decisively; Phase 2 student failed this stress bar even while beating packet2 |
| Projection-student `DS` reopen bar | `<= 8 / <= 10 / <= 100 / <= 1e-6` | Projected-student validation winner | Failed; `DS` remains blocked |

### What the Results Mean

- Packet2 remains the best deployed admissible-surface baseline because it was the strongest direct-replacement winner under real deployment, not just under oracle routing.
- Packet3 was scientifically important even though it was not deployable: the oracle beat packet2, so branch-local information matters. But because predicted routing lost to packet2 and oracle still missed the bar, the line still needed both routing and edge-local regression to improve.
- Packet4 existed to remove the packet3 argmax discontinuity without abandoning admissibility or the future autodiff path. It failed because:
  - the deployed winner still lost to packet2 on broad plastic MAE, hard plastic MAE, and hard p95 principal;
  - it did not clear the `25 / 28 / 250 / 1e-6` continue bar;
  - its oracle diagnostic was worse than its deployed routing, so packet4 did not earn routing-gap rescue credit.
- Program X existed because packet4 stopped the surface-coordinate family but did not yet mathematically disprove direct replacement. Program X asked the narrowest remaining question: if the true branch is known and the model predicts the exact branchwise latent variables already used by the constitutive formulas, does the line become credible?
- Program X answered that question negatively. Stage 0 exactness proved the dataset, latent extraction, and decode were numerically correct. The failure therefore sits with learnability of the branchwise exact-latent targets on the frozen benchmark, not with the audit setup.
- The March 26 strategic override changed the question after Program X. The projection-student line no longer asked the network to learn the admissible manifold directly. Instead it asked whether a strong stress predictor could be preserved by projecting plastic principal stress onto the exact admissible set.
- Phase 0 answered that question positively. The exact projection layer collapsed yield tails from about `1e-1` to about `2e-8` on both validation and held-out test while keeping stress metrics close to the raw-teacher anchor.
- Phase 2 answered the next question only partially. A bounded projected student beat packet2, packet3 oracle, and packet4 on the primary validation stress metrics while preserving admissibility, so the line is scientifically alive. But because the best student remained far worse than the projected teacher and missed the `<= 8 / <= 10 / <= 100 / <= 1e-6` `DS` bar, the forward model is still not tangent-ready.
- Final decision state:
  - keep packet2 / packet3 / packet4 / Program X closed as historical families;
  - continue only the `projection-student` line from the exact ordered-principal projection setup;
  - do not reopen packet5, new routing, new atlas, or exact-latent follow-ons from this state;
  - keep `DS` blocked until the stronger projection-student bar is met.

## Current Repository State

- Canonical closeout set:
  - `docs/repository_general_report_20260326.md`
  - `docs/repository_general_report_metric_dictionary_20260326.md`
  - `docs/cover_layer_comprehensive_report_20260326.md`
  - `docs/repository_general_report_appendix_20260326.md`
  - `docs/repository_general_report_processed_manifest_20260326.json`
  - `progress.md`
- Paired projection-student work-packet docs:
  - `docs/tasks/projection_student_work_packet_20260326.md`
  - `docs/executions/projection_student_work_packet_20260326.md`
- Last executed program: projection-student reopen audit and bounded projected-student run.
- Best retained packet-family direct baseline: packet2 deployed winner.
- Best admissibility-preserving reopen audit: Phase 0 projected teacher.
- Best learned reopen candidate: Phase 2 `medium_exact` projected student.
- Current state: `continue_projection_student`; `DS` blocked.
- Raw evidence is still preserved in `experiment_runs/**/*.json`, `experiment_runs/**/*.csv`, and the retained background / prehistory markdown families.

## Remaining Open Questions

- Packet3 oracle held-out test metrics remain unavailable in canonical raw form. This report deliberately does not backfill them from prose.
- The high-level synthetic, real-sim, per-material, and local cover-layer summary docs have now been absorbed here and deleted when safe. The remaining cover-layer markdown prehistory has also been collapsed into `docs/cover_layer_comprehensive_report_20260326.md`, so future readers do not need the old per-run cover-layer notes.
- The projection-student route is now the only live direct-replacement continuation. Any work outside that route still requires a new explicit strategic brief rather than continuation from the retired packet or Program X families.

## Source Digest Map

| Source class | Absorbed into this report | Cleanup outcome |
| --- | --- | --- |
| Root handoff prompts: `next_agent_prompt_*.md` | Timeline, decision logic, source digest context | Deleted after manifest validation |
| Root strategic briefs: `report2.md`-`report11.md` | Timeline, benchmark history, decision bars, stop/continue logic | Deleted after manifest validation |
| Hybrid family docs: `docs/hybrid_pivot_*.md`, `docs/hybrid_gate_redesign_*.md`, `docs/return_mapping_accel_*.md`, `docs/expert_complete_report_20260324.md` | Timeline and benchmark transition | Deleted after manifest validation |
| Historical baseline summary docs: `docs/demo_training_experiments.md`, `docs/high_capacity_study.md`, `docs/real_sim_validation.md`, `docs/staged_real_sim_training.md`, `docs/relative_error_visual_report.md`, `docs/per_material_*.md`, `docs/cover_layer_cyclic_sweep.md`, `docs/cover_layer_fitted_refresh_heavy_w1024_d6.md`, `docs/cover_layer_tail_safety_*.md`, `docs/cover_layer_best_candidate_model_card.md`, `docs/next_workplan_from_report.md` | Timeline, historical baseline condensation, source digest context | Deleted after second-pass manifest validation |
| Cover-layer raw markdown stack: `docs/cover_layer*.md`, `docs/_smoke_heavy_campaign.md` | Dedicated local cover-layer closeout, historical baseline condensation, source digest context | Deleted after third-pass cover-layer consolidation into `docs/cover_layer_comprehensive_report_20260326.md` |
| Direct-replacement packet docs: `docs/nn_replacement_abr_*.md`, `docs/nn_replacement_surface_*.md`, `docs/expert_current_state_report_20260324.md`, `docs/nn_replacement_surface_packet3_*.md`, `docs/nn_replacement_surface_packet4_*.md` | Packet-level results, normalized tables, packet4 stop logic | Deleted after manifest validation |
| Program X docs and comparison memos: `docs/nn_replacement_exact_latent_*.md`, `experiment_runs/.../execution_plan.md`, `experiment_runs/.../programx_vs_packet2_packet3_packet4_val.md`, `experiment_runs/.../packet4_vs_packet2_packet3_val.md` | Program X exactness, branchwise winners, terminal stop logic | Deleted after manifest validation |
| Retained background / theory / active packet docs: `README.md`, `sampling.md`, `report.md`, `tasks1.md`, `docs/mohr_coulomb.md`, `docs/cover_layer_comprehensive_report_20260326.md`, `docs/tasks/projection_student_work_packet_20260326.md`, `docs/executions/projection_student_work_packet_20260326.md`, the retained `docs/**/*.md` families cataloged in `docs/repository_general_report_appendix_20260326.md`, smoke/heavy campaign phase reports under `experiment_runs/` | Background context, dedicated local cover-layer closeout, the active projection-student work-packet pair, and retained non-cover-layer prehistory | Retained |

Manifest summary for this consolidation:

- Safe-deleted markdown inputs: see `deleted_files` in `docs/repository_general_report_processed_manifest_20260326.json`
- Intentionally retained markdown inputs: see `retained_files`, `retained_because`, and `family_inventory` in `docs/repository_general_report_processed_manifest_20260326.json`
- Retained `docs/**/*.md` family catalog: see `docs/repository_general_report_appendix_20260326.md`
