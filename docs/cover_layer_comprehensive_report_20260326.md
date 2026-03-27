# Cover-Layer Comprehensive Report 20260326

This report consolidates the entire remaining `cover_layer*.md` stack plus `_smoke_heavy_campaign.md` into one closeout.

Scope of this cleanup:

- absorbed markdown inputs: `65`
- families absorbed:
  - branch-predictor prehistory: `39`
  - generator / coverage / B-copula: `19`
  - structured local-surrogate / routing: `7`

The raw experiment artifacts remain under `experiment_runs/real_sim/**`. This report replaces the per-run markdown layer, not the raw evidence.

## Executive Summary

- The cover-layer work produced three real lessons, not one repo-wide control model.
- Best local branch-classifier result: the expert-principal hierarchical branch predictor reached about `0.9007` real-test accuracy and `0.8916` real-test macro recall on the fixed `4`-call representative branch slice after the generator moved into principal-space real-like sampling.
- Best local single-model stress surrogate: fitted-refresh `cover_raw_branch_w1024_d6_local_noise` reached real-test stress MAE `1.4951`, RMSE `9.7174`, and branch accuracy `0.9577` on the cover-layer exact-domain holdout.
- Best local routed candidate: `edge_hard_mined_gate_raw_threshold_t0.65` reached real MAE `2.8081`, RMSE `16.0203`, branch accuracy `0.9559`, and p99 relative error `1.5390`, but it relied on holdout-informed hard mining and was never safe to promote as a general constitutive replacement.
- Main scientific conclusion: domain coverage and branch geometry mattered more than brute model capacity. Generator design was the decisive lever. The hardest unresolved region stayed the real `smooth / right_edge` transition.
- Main strategic conclusion: the cover-layer line was valuable local prehistory, but it was too local, too benchmark-specific, and in the routed case too holdout-informed to become the controlling repository benchmark. Its durable output was insight, not the final repo decision rule.

## Evaluation Scopes And Comparability

The cover-layer stack used multiple local benchmarks. These results should not be compared directly to the later March 24 packet benchmark without that caveat.

### 1. Branch-predictor diagnostic benchmark

- material scope: `cover_layer` only
- training regime: synthetic-only
- real evaluation regime: fixed representative slice from the full export
- split shape:
  - `generator_fit`: `332` calls
  - `real_val`: `111` calls
  - `real_test`: `111` calls
- practical diagnostic slice used by the best branch-classifier runs:
  - `4` real validation calls
  - `4` real test calls
  - up to `128` elements per call
  - `512` elements, `5632` integration-point labels per slice

Interpretation:
These metrics are good for controlled branch-transfer iteration, but they are not full-population deployment numbers.

### 2. Cover-layer exact-domain stress-surrogate benchmark

- material scope: `cover_layer` only
- exact-domain real dataset path: `experiment_runs/real_sim/cover_layer_single_material_20260313/cover_layer_full_real_exact_256.h5`
- canonical exact-domain local test size used by the cyclic / adaptive / fitted-refresh line:
  - primary test: `7967`
  - cross test: `3922`

Interpretation:
These are real local stress-regression benchmarks for one material family, not the later grouped-call multi-material packet benchmark.

### 3. Routed-expert / tail-safety benchmark

- control baseline: `baseline_raw_branch`
- local holdout: same cover-layer exact-domain split
- special caveat: the strongest routed candidates used hard-mined edge experts informed by holdout difficulty

Interpretation:
Useful as a shadow-solver local candidate. Not valid as a general clean deployment winner.

## Program Map

| Family | Purpose | Canonical evidence | Key result | Terminal conclusion |
| --- | --- | --- | --- | --- |
| Exact-domain cyclic baselines | Establish whether a cover-layer-only real exact-domain surrogate is numerically credible | `experiment_runs/real_sim/cover_layer_cyclic_20260312/sweep_summary.json` | Best exact-domain single-model baseline reached primary MAE `4.7124` and cross RMSE `11.6177-11.7293` depending on width/depth winner | Exact-domain training was credible, but not yet the strongest local route |
| Adaptive exact-domain continuation | Test whether longer adaptive scheduling beats the cyclic baseline | `experiment_runs/real_sim/cover_layer_adaptive_20260312_lr2p5e4/cover_raw_branch_w1024_d6_adaptive/summary.json` | Adaptive `1024x6` degraded to primary MAE `5.6846`, RMSE `21.0031` | More schedule/capacity alone was not the answer |
| Epoch-refresh and fitted-refresh generator line | Replace mismatched synthetic training with distributions fitted to the real cover-layer state cloud | `experiment_runs/real_sim/cover_layer_epoch_refresh_20260312/distribution_summary.json`; fitted-refresh summaries | Naive epoch-refresh collapsed on real data, but fitted-refresh later produced the strongest single-model local result | Generator fidelity mattered more than capacity |
| Single-material structured regression line | Test structured residual and hybrid `U/B` augmentation against a direct raw baseline | `experiment_runs/real_sim/cover_layer_single_material_20260313/*`; `experiment_runs/real_sim/cover_layer_principal_correction_20260313/*` | Only `baseline_raw_branch` remained numerically stable; structured residuals failed badly | Do not spend time on that residual target family |
| Branch experts and routed ensemble line | Exploit branch-local stress signal with separate plastic experts | `experiment_runs/real_sim/cover_layer_branch_experts_20260313/*`; gate/tail-safety summaries | Oracle experts improved MAE, but predicted routing initially lost; later gated and hard-mined routes became strong local candidates | Gate quality was the blocker; routed winner stayed local and caveated |
| Strain-only and element-state branch predictors | Test weaker representations for branch classification | strain-only summaries; element-branch summaries | Both lines transferred poorly even when synthetic metrics were strong | Reduced material and trial-stress context were necessary |
| Pointwise constitutive branch-predictor line | Predict branch from strain, trial stress, and reduced material with synthetic-only training | branch-predictor summaries from March 14-16 | Moved from failed transfer to about `90%` real branch accuracy after principal-space generator redesign | Valuable geometry/failure insight; still not a full constitutive replacement |
| Generator coverage / smooth-focus diagnostics | Determine whether transfer failure was support mismatch, wrong local geometry, or both | generator baseline, screen, correction, failure analyses | Coverage fixes helped sharply, but smooth-focus anchoring still did not rescue real `smooth` | Remaining blocker was local branch geometry, not just missing counts |
| FE-local `B`-copula generator line | Test FE-local displacement sampling instead of direct strain perturbations | `experiment_runs/real_sim/cover_layer_b_copula_stats_20260312/branch_raw/summary.json`; `experiment_runs/real_sim/cover_layer_b_copula_20260312/cover_raw_branch_w384_d6_branch_raw/train.log` | Screened branch counts looked promising, but training stayed noncompetitive | Do not continue the current global B-copula fit |

## Normalized Result Tables

### Branch-Predictor Progression

All rows below use the fixed local branch slice, not the March 24 packet benchmark.

| Phase | Canonical run | Real test acc | Real test macro | Real `smooth` recall | Synthetic acc | Synthetic macro | Main takeaway |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Early constitutive-feature baseline | `cover_layer_branch_predictor_baseline_20260314` | `0.7031` | `0.4005` | `0.0000` | `0.6260` | `0.5019` | Basic pointwise structure still failed badly on real branch balance |
| Corrected staged local-noise generator | `cover_layer_branch_predictor_staged_seeded_balanced_20260314` | `0.6000` | `0.5741` | `0.0051` | `0.9763` | `0.9827` | Synthetic task became easy, but real `smooth` still collapsed |
| Expert-principal generator breakthrough | `cover_layer_branch_predictor_expert_principal_20260315` | `0.8865` | `0.8750` | `0.7490` | `0.9572` | `0.9576` | Domain coverage redesign was the decisive breakthrough |
| Larger long run | `cover_layer_branch_predictor_expert_principal_w1024_d6_long_20260315` | `0.8928` | `0.8828` | `0.7500` | `0.9658` | `0.9655` | Capacity/training helped, but much less than the generator redesign |
| Heavy post-train campaign best internal checkpoint | `cover_layer_branch_predictor_heavy_campaign_20260316` | `0.9007` | `0.8916` | `0.7884` | `0.9656` core val | `0.9387` hard val | The line plateaued around `90%`/`89%`; optimization was no longer the main blocker |

Additional heavy-campaign diagnostics:

- full real-validation accuracy: `0.8958`
- full real-validation macro recall: `0.9008`
- hard real-validation macro recall: `0.8943`
- the main remaining harmful confusions stayed `smooth <-> edge` and `edge <-> apex`

### Rejected Branch-Predictor Variants

| Variant family | Best canonical result | Key numbers | Why it lost |
| --- | --- | --- | --- |
| Strain-only pointwise predictors | `cover_layer_strain_branch_predictor_synth_only_w256_d3_lrcycle_20260314` | real test acc `0.4672`, macro `0.5019`, elastic recall `0.0000`, synthetic acc `0.9264` | Strain alone was structurally ambiguous once reduced material varied along SSR |
| Direct element-state predictors | `cover_layer_element_branch_predictor_smoke_20260314` | real test acc `0.6042`, macro `0.3801` | Element-state raw inputs did not generalize even before harder structured follow-up |
| Element synthetic mastery | `gate_a_w256_d6` / `gate_b_w256_d6` | synthetic accuracy `0.9968`, macro `0.9978`; real-like gate-B accuracy `0.3080` | Synthetic mastery did not transfer to real-like branch geometry |
| Structured element follow-up | `b_diagnostics_summary.json` | real test accuracy only `0.2408-0.2665` across follow-ups | More seed reshuffling and call expansion did not rescue the element-state route |

### Generator And Coverage Diagnostics

| Diagnostic stage | Canonical source | Key normalized numbers | Reading |
| --- | --- | --- | --- |
| Early generator baseline | `cover_layer_branch_generator_baseline_20260314/summary.json` | real strain norm mean / p95 `1.3059 / 5.8541`; synthetic `0.0485 / 0.0907` | The old synthetic deformation scale was wildly off |
| Generator screen | `cover_layer_branch_generator_screen_20260314/summary.json` | best screened mode `empirical_local_noise`; branch TV `0.3568`; strain-mean rel err `0.1773` | First useful FE-local generator, but still branch-misaligned |
| Corrected branch-balanced local generator | `cover_layer_branch_generator_correction_20260314/coverage_summary.json` | branch TV `0.3009 -> 0.2663`; mean strain rel err `0.1572 -> 0.0102`; p95 rel err worsened to `0.1929` | Coverage got tighter on average but still missed the upper tail |
| Failed-element support audit | `failed_element_support_analysis.json` plus note | failed median seed-neighborhood `z_rms 4.3868` vs success `0.1436`; failed fraction outside synthetic-seed p95 `0.7733` | Many failed real elements were outside practical generator support |
| Smooth-focus retry | smooth-focus run summaries | real test acc `0.6000 -> 0.5891` or `0.5447`; real `smooth` recall still about `0.004-0.007` | Nearby support alone did not fix the `smooth` failure |

Interpretation:

- Coverage mismatch was real and fixing it helped a lot.
- The remaining blocker was not just missing volume in state space.
- Real `smooth` points still had the wrong local geometry relative to the synthetic neighborhoods.

### Local Stress-Surrogate Progression

| Family | Canonical run | Real test MAE | Real test RMSE | Branch acc | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| Exact-domain cyclic baseline, best primary MAE | `cover_raw_branch_w384_d6` | `4.7124` | `13.1843` | `0.9195` | Best primary exact-domain cyclic baseline |
| Exact-domain cyclic baseline, best cross RMSE | `cover_raw_branch_w512_d6` | cross MAE `4.5254` | cross RMSE `11.6177` | cross acc `0.9182` | Best cross-split cyclic RMSE |
| Adaptive exact-domain continuation | `cover_raw_branch_w1024_d6_adaptive` | `5.6846` | `21.0031` | `0.9007` | Longer schedule was worse than cyclic |
| Epoch-refresh synthetic retraining | `cover_raw_branch_w384_d6_epoch_refresh` | `23.1082` | `53.8699` | `0.3983` | Refreshing the wrong synthetic distribution collapsed on real data |
| Fitted-refresh heavy | `cover_raw_branch_w384_d6_local_noise` | `1.8579` | `10.3853` | `0.9498` | Matching the real distribution beat exact-domain cyclic training |
| Fitted-refresh heavy, widened winner | `cover_raw_branch_w1024_d6_local_noise` | `1.4951` | `9.7174` | `0.9577` | Best single-model local cover-layer result in the repo |

Best fitted-refresh `w1024 d6` per-branch MAE:

- elastic: `0.5675`
- smooth: `1.0963`
- left_edge: `2.3914`
- right_edge: `2.8533`
- apex: `0.2908`

### Structured Local-Surrogate And Routing Results

| Stage | Canonical source | Real MAE | Real RMSE | Branch acc | Main takeaway |
| --- | --- | ---: | ---: | ---: | --- |
| Direct local baseline | `baseline_raw_branch` | `8.2331` | `22.9041` | `0.8255` | Only stable production-grade model in the early single-material study |
| Principal correction, real-only | `principal_trial_branch_residual_real` | `146.7714` | `262.0139` | `0.7207` | Corrected principal residual still failed badly |
| Principal correction, hybrid | `principal_trial_branch_residual_hybrid` | `127.8092` | `223.6647` | `0.7118` | Hybrid helped only relative to a broken family, not the baseline |
| Oracle branch experts | `oracle_branch_experts` | `7.3426` | `31.0570` | `1.0000` | Branch-local stress maps were learnable |
| Predicted branch experts | `predicted_branch_experts` | `8.7657` | `33.3613` | `0.8255` | The gate was the blocker |
| Best gate without hard mining | `gate_trial_threshold_t0.85` | `7.1258` | `29.1589` | `0.9433` | Better MAE, but still tail-heavy and not a clean overall win |
| Best local routed candidate | `edge_hard_mined_gate_raw_threshold_t0.65` | `2.8081` | `16.0203` | `0.9559` | Strong local route, but caveated by holdout-informed hard mining |

Best routed-candidate details:

- p99 relative error: `1.5390`
- edge combined MAE: `3.2216`
- top-bin sample MAE: `36.3578`

This was good enough for a solver shadow-test candidate, not for a clean constitutive replacement claim.

### FE-Local B-Copula Line

| Stage | Canonical source | Key result | Conclusion |
| --- | --- | --- | --- |
| Screening | `experiment_runs/real_sim/cover_layer_b_copula_stats_20260312/branch_raw/summary.json` | `branch_raw` was the best screened mode with total fit score about `2.27-2.34` and branch TV about `0.011`, but principal/stress tail log-errors remained large | Good branch counts alone were not enough |
| Training attempt | `experiment_runs/real_sim/cover_layer_b_copula_20260312/cover_raw_branch_w384_d6_branch_raw/train.log` plus preserved evaluation note | real test stress MAE `48.42`, RMSE `88.71`, branch acc `0.440`; synthetic stress MAE `39.49`, RMSE `69.03`, branch acc `0.916` | Do not continue the current global B-copula fit |

Interpretation:
The FE `B` operator route itself was not rejected. The rejected part was the overly crude global copula fit at the wrong level of structure.

## Cross-Cutting Findings

1. Domain coverage dominated capacity.
   The decisive jump came from the expert-inspired principal-space generator, not from widening networks. The main transition was roughly `0.7058 -> 0.8865` real-test branch accuracy when sampling changed, then only `0.8865 -> 0.8928` from scaling/training.

2. Reduced material and trial stress were mandatory context.
   Strain-only branch predictors were not just weaker; they were structurally ambiguous once the reduced constitutive state varied along the SSR path.

3. Branch-local stress signal was real, but routing was the bottleneck.
   Oracle branch experts beat the direct baseline, while predicted routing lost. That same pattern later reappeared at the repository level in packet3: local branch structure helped, but deployed routing was brittle.

4. `smooth` was the central failure region.
   The corrected staged generator achieved synthetic macro recall `0.9827` while real `smooth` recall stayed `0.0051`. Even smooth-focused anchoring did not fix it. The line learned synthetic `smooth`, not real `smooth`.

5. “Near the cloud” was not enough.
   Failed real elements were often outside the generator’s practical seed neighborhoods but not always far from the synthetic strain cloud. That means the remaining issue was local branch geometry and label structure, not only coarse global support.

6. Matching the real training distribution beat exact-domain purity.
   Exact-domain cyclic models were respectable. Naive epoch-refresh was disastrous. Fitted-refresh, which better matched the real cover-layer state distribution, produced the strongest single-model local result.

7. The best routed local candidate was real but not clean.
   Hard-mined edge experts made the local route strong enough for shadow testing, but the use of holdout-informed mining prevented it from becoming a general trusted control.

## Final Disposition

- Keep the cover-layer work as historical local prehistory and methodology insight.
- Treat the strongest durable local winners as:
  - branch classification: heavy expert-principal branch predictor
  - single-model stress surrogate: fitted-refresh `w1024 d6 local_noise`
  - routed local shadow candidate: `edge_hard_mined_gate_raw_threshold_t0.65`
- Do not reinterpret those wins as repository-wide deployment winners.
- The line did not become the controlling benchmark because:
  - it was material-local;
  - branch classification used a representative diagnostic slice rather than the later grouped-call benchmark;
  - routed tail-safety success relied on holdout-informed hard mining;
  - later repo control shifted to the March 24 packet benchmark and then Program X.

The durable value of this stack was that it established several principles later repo work reused:

- data regime matters more than synthetic convenience;
- structural exactness alone is not enough if the learned representation misses the hard plastic geometry;
- local branch/oracle gains are scientifically interesting but not sufficient for deployment unless the deployed route also wins.

## Source Digest And Cleanup

Absorbed and superseded families:

- `docs/cover_layer_branch_predictor_*.md`
- `docs/cover_layer_strain_branch_predictor_*.md`
- `docs/cover_layer_element_branch_*.md`
- `docs/cover_layer_branch_generator_correction_report.md`
- `docs/cover_layer_branch_smooth_focus_report.md`
- `docs/cover_layer_domain_coverage_brief.md`
- `docs/cover_layer_failed_element_support_report.md`
- `docs/cover_layer_sampling_expert_opinion_test.md`
- `docs/cover_layer_epoch_refresh*.md`
- `docs/cover_layer_fitted_refresh*.md`
- `docs/cover_layer_b_copula*.md`
- `docs/cover_layer_single_material_*.md`
- `docs/cover_layer_adaptive_w1024_d6_lr2p5e4.md`
- `docs/cover_layer_principal_correction.md`
- `docs/cover_layer_branch_experts.md`
- `docs/cover_layer_gate_experiments.md`
- `docs/_smoke_heavy_campaign.md`

Retained instead of deleted:

- raw artifacts in `experiment_runs/real_sim/**`
- code and theory files outside the cover-layer markdown stack

Cleanup rule applied here:

- keep raw artifact provenance
- keep the new condensed closeout
- remove the redundant markdown layer once the repo-level manifest is updated
