# Per-Material Synthetic-to-Real Study

This report tests a narrower version of the surrogate problem:
- split the real sampled data by material family,
- train one surrogate per material family,
- train on exact synthetic labels generated from the real train-split input domain,
- evaluate only on held-out real test data.

This is not broad synthetic sampling. It is an empirical-domain synthetic setup: the training inputs come
from the real train split of each material family, but the labels are recomputed with the exact Python
constitutive operator. That directly tests whether the mixed-material global model was the main bottleneck.

Primary real dataset: `experiment_runs/real_sim/train_sample_512.h5`
Cross real dataset: `experiment_runs/real_sim/baseline_sample_256.h5`
Global baseline checkpoint: `experiment_runs/real_sim/staged_20260312/rb_staged_w512_d6/best.pt`

## Material Mapping

| Material | Train | Val | Test | Train SRF q05-q95 | Train max | Branch train counts |
|---|---:|---:|---:|---|---:|---|
| general_foundation | 16563 | 2039 | 2145 | 1.100 - 1.666 | 2.4791e-02 | {"elastic": 6689, "smooth": 9155, "left_edge": 719, "right_edge": 0, "apex": 0} |
| weak_foundation | 32351 | 4066 | 3928 | 1.100 - 1.666 | 2.0537e-01 | {"elastic": 4832, "smooth": 17598, "left_edge": 7806, "right_edge": 2046, "apex": 69} |
| general_slope | 114661 | 14470 | 14325 | 1.100 - 1.666 | 2.5124e+01 | {"elastic": 16920, "smooth": 45518, "left_edge": 24012, "right_edge": 15422, "apex": 12789} |
| cover_layer | 63343 | 7790 | 7967 | 1.100 - 1.666 | 3.6412e+01 | {"elastic": 9330, "smooth": 16484, "left_edge": 15452, "right_edge": 10959, "apex": 11118} |

## Primary Real Test

| Material | Samples | MAE | RMSE | Max Abs | Branch Acc |
|---|---:|---:|---:|---:|---:|
| general_foundation | 2145 | 0.9729 | 1.6418 | 24.7492 | 0.9911 |
| weak_foundation | 3928 | 0.9444 | 1.7276 | 51.2800 | 0.9824 |
| general_slope | 14325 | 19.4763 | 33.8573 | 704.9789 | 0.9228 |
| cover_layer | 7967 | 12.9654 | 24.1759 | 503.0659 | 0.8637 |
| aggregate | 28365 | 13.6820 | 27.2709 | 704.9789 | 0.9196 |

Global baseline on the same per-material primary test splits:

| Material | Samples | MAE | RMSE | Max Abs | Branch Acc |
|---|---:|---:|---:|---:|---:|
| general_foundation | 2145 | 1.8580 | 3.2566 | 34.5765 | 0.9585 |
| weak_foundation | 3928 | 3.6945 | 8.3429 | 209.1459 | 0.9539 |
| general_slope | 14325 | 5.1771 | 19.9995 | 936.4612 | 0.9564 |
| cover_layer | 7967 | 5.6771 | 19.0208 | 531.1244 | 0.9072 |
| aggregate | 28365 | 4.8612 | 17.7217 | 936.4612 | 0.9424 |

## Cross Real Test

| Material | Samples | MAE | RMSE | Max Abs | Branch Acc |
|---|---:|---:|---:|---:|---:|
| general_foundation | 1053 | 0.9782 | 1.6597 | 24.3849 | 0.9905 |
| weak_foundation | 1957 | 1.0442 | 2.8963 | 144.7010 | 0.9770 |
| general_slope | 7251 | 20.1161 | 35.9528 | 1049.0601 | 0.9291 |
| cover_layer | 3922 | 12.8657 | 24.1083 | 475.4662 | 0.8562 |
| aggregate | 14183 | 14.0587 | 28.6866 | 1049.0601 | 0.9201 |

Global baseline on the same per-material cross test splits:

| Material | Samples | MAE | RMSE | Max Abs | Branch Acc |
|---|---:|---:|---:|---:|---:|
| general_foundation | 1053 | 1.8040 | 3.1973 | 35.9948 | 0.9544 |
| weak_foundation | 1957 | 4.0171 | 10.4822 | 311.4559 | 0.9520 |
| general_slope | 7251 | 5.0530 | 19.5313 | 759.1741 | 0.9588 |
| cover_layer | 3922 | 5.3597 | 18.2686 | 473.3656 | 0.9072 |
| aggregate | 14183 | 4.7536 | 17.4137 | 759.1741 | 0.9432 |

## Notes

- If the per-material synthetic models win, the mixed-material coupling was a real part of the error budget.
- If they still lose badly to the real-trained global model, then the dominant issue is not material mixing, but state-distribution and architecture bias.
- The `material_fit_error_max` values should stay very small; if they are not, the material-family reconstruction is suspect.
