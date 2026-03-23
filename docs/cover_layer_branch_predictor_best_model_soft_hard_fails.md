# Best Branch Model: Soft vs Hard Fail Report

## Model

- checkpoint: `experiment_runs/real_sim/cover_layer_branch_predictor_expert_principal_w1024_d6_lbfgs_100cycles_20260315/loop_17_accepted.pt`
- feature set: `trial_raw_material`
- architecture: `hierarchical w1024 d6`
- hard-fail threshold `tau`: `0.01`

## Split Summary

| split | accuracy | macro recall | wrong | soft fail | hard fail | hard adjacent | hard non-adjacent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| real_val_slice | 0.8825 | 0.8787 | 0.1175 | 0.0108 | 0.1067 | 0.0971 | 0.0096 |
| real_val_large | 0.9045 | 0.9082 | 0.0955 | 0.0075 | 0.0880 | 0.0812 | 0.0068 |
| real_val_hard | 0.8763 | 0.8943 | 0.1237 | 0.0093 | 0.1144 | 0.1053 | 0.0091 |
| real_val_full | 0.8958 | 0.9008 | 0.1042 | 0.0069 | 0.0973 | 0.0900 | 0.0073 |
| real_test | 0.8981 | 0.8905 | 0.1019 | 0.0055 | 0.0964 | 0.0890 | 0.0075 |

## Interpretation

- `real_val_large`: soft among wrong = `0.0785`, hard among wrong = `0.9215`, median / p95 `rel_e_sigma` on wrong = `0.2257` / `2.8090`
- `real_val_hard`: soft among wrong = `0.0752`, hard among wrong = `0.9248`, median / p95 `rel_e_sigma` on wrong = `0.2402` / `2.9038`
- `real_val_full`: soft among wrong = `0.0658`, hard among wrong = `0.9342`, median / p95 `rel_e_sigma` on wrong = `0.2394` / `3.0521`
- `real_test`: soft among wrong = `0.0540`, hard among wrong = `0.9460`, median / p95 `rel_e_sigma` on wrong = `0.2843` / `2.9087`

## Top Hard Confusions: real_val_large

| true | predicted | count | hard rate | soft rate | median rel_e_sigma | p95 rel_e_sigma |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| left_edge | right_edge | 133 | 1.0000 | 0.0000 | 0.4751 | 2.7946 |
| right_edge | left_edge | 130 | 1.0000 | 0.0000 | 0.5759 | 3.6515 |
| smooth | apex | 35 | 1.0000 | 0.0000 | 1.2856 | 1.5942 |
| apex | smooth | 10 | 1.0000 | 0.0000 | 1.0860 | 1.4454 |
| right_edge | apex | 466 | 0.9893 | 0.0107 | 1.0872 | 1.6718 |
| left_edge | apex | 398 | 0.9874 | 0.0126 | 1.1601 | 1.5468 |
| apex | right_edge | 378 | 0.9868 | 0.0132 | 0.8615 | 7.6823 |
| apex | left_edge | 509 | 0.9686 | 0.0314 | 0.5685 | 4.2085 |
| right_edge | smooth | 361 | 0.9141 | 0.0859 | 0.1207 | 2.5698 |
| smooth | left_edge | 826 | 0.9116 | 0.0884 | 0.0875 | 0.3552 |
| smooth | right_edge | 671 | 0.9106 | 0.0894 | 0.0905 | 0.3181 |
| left_edge | smooth | 241 | 0.8465 | 0.1535 | 0.0617 | 1.3695 |

## Top Hard Confusions: real_test

| true | predicted | count | hard rate | soft rate | median rel_e_sigma | p95 rel_e_sigma |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| apex | left_edge | 61 | 1.0000 | 0.0000 | 0.5867 | 5.1011 |
| right_edge | apex | 36 | 1.0000 | 0.0000 | 1.0517 | 1.6783 |
| left_edge | right_edge | 20 | 1.0000 | 0.0000 | 0.8747 | 6.8033 |
| smooth | apex | 14 | 1.0000 | 0.0000 | 1.1368 | 1.4906 |
| right_edge | left_edge | 8 | 1.0000 | 0.0000 | 0.3910 | 1.1217 |
| elastic | smooth | 1 | 1.0000 | 0.0000 | 0.0226 | 0.0226 |
| left_edge | apex | 94 | 0.9894 | 0.0106 | 1.2690 | 1.5314 |
| apex | right_edge | 61 | 0.9836 | 0.0164 | 0.8456 | 9.2894 |
| smooth | right_edge | 53 | 0.9434 | 0.0566 | 0.1023 | 0.3165 |
| right_edge | smooth | 64 | 0.9375 | 0.0625 | 0.2056 | 4.4474 |
| smooth | left_edge | 109 | 0.9083 | 0.0917 | 0.0924 | 0.3297 |
| left_edge | smooth | 30 | 0.8667 | 0.1333 | 0.0528 | 1.1094 |