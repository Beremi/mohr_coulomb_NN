# Cover Layer Branch Predictor Heavy Post-Train Campaign

## Summary

- base architecture: `hierarchical w2048 d6`
- features: `trial_raw_material`
- generator: `expert principal hybrid`
- selection policy: multi-track (`best_real_val`, `best_balanced`, `best_synthetic`, `best_internal_step`)

## Comparison

| checkpoint | real val macro | real test macro | core val acc | hard val macro |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.8787 | 0.8916 | 0.9656 | 0.9387 |
| phase1_best_real_val | 0.8789 | 0.8906 | 0.9656 | 0.9390 |
| phase1_best_balanced | 0.8789 | 0.8906 | 0.9656 | 0.9390 |
| phase2_best_real_val | 0.8784 | 0.8912 | 0.9656 | 0.9392 |
| phase2_best_balanced | 0.8784 | 0.8912 | 0.9656 | 0.9392 |

## Deliverables

- final_best_real_val: `experiment_runs/real_sim/_smoke_heavy_campaign/checkpoints/phase2_best_real_val.pt`
- final_best_balanced: `experiment_runs/real_sim/_smoke_heavy_campaign/checkpoints/phase2_best_balanced.pt`
- final_best_synthetic: `experiment_runs/real_sim/_smoke_heavy_campaign/checkpoints/phase2_best_synthetic.pt`

## Recommendation

- deployment: `experiment_runs/real_sim/_smoke_heavy_campaign/checkpoints/phase2_best_real_val.pt`
- balanced continuation anchor: `experiment_runs/real_sim/_smoke_heavy_campaign/checkpoints/phase2_best_balanced.pt`
- synthetic-faithful analysis: `experiment_runs/real_sim/_smoke_heavy_campaign/checkpoints/phase2_best_synthetic.pt`

## Curves

![Campaign history](../experiment_runs/real_sim/_smoke_heavy_campaign/campaign_history.png)
