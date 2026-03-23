# Cover Layer Direct Element Branch Predictor: Synthetic-Domain Mastery

## Summary

This report focuses on the **direct one-material raw-element model**:

- input: canonicalized `coords(10x3) + disp(10x3)`
- output: `11 x 5` branch logits
- training objective for this phase: **master the synthetic domain first**
- real validation/test are tracked only as diagnostics

## Frozen Synthetic Benchmark

- synthetic train seed calls: `24`
- synthetic eval seed calls: `8`
- synthetic IID val/test: `4096 / 8192`
- synthetic hard val/test: `4096 / 8192`
- fixed hard benchmark noise scale: `0.35`

![Branch benchmark frequencies](../experiment_runs/real_sim/cover_layer_element_branch_synth_mastery_20260314/benchmark_branch_frequencies.png)

Benchmark branch frequencies:

```json
{
  "train_seed_real": {
    "elastic": 0.1542376893939394,
    "smooth": 0.24591619318181818,
    "left_edge": 0.2421579071969697,
    "right_edge": 0.1933297821969697,
    "apex": 0.16435842803030304
  },
  "synthetic_iid_val": {
    "elastic": 0.030917080965909092,
    "smooth": 0.3053311434659091,
    "left_edge": 0.27274946732954547,
    "right_edge": 0.07939009232954546,
    "apex": 0.3116122159090909
  },
  "synthetic_iid_test": {
    "elastic": 0.030728426846590908,
    "smooth": 0.30308948863636365,
    "left_edge": 0.27400346235795453,
    "right_edge": 0.080810546875,
    "apex": 0.3113680752840909
  },
  "synthetic_hard_val": {
    "elastic": 0.03329190340909091,
    "smooth": 0.3182040127840909,
    "left_edge": 0.2678888494318182,
    "right_edge": 0.07570578835227272,
    "apex": 0.3049094460227273
  },
  "synthetic_hard_test": {
    "elastic": 0.031205610795454544,
    "smooth": 0.31828169389204547,
    "left_edge": 0.2709738991477273,
    "right_edge": 0.07387473366477272,
    "apex": 0.3056640625
  },
  "real_val": {
    "elastic": 0.23206676136363635,
    "smooth": 0.21413352272727273,
    "left_edge": 0.2176846590909091,
    "right_edge": 0.17897727272727273,
    "apex": 0.1571377840909091
  },
  "real_test": {
    "elastic": 0.2549715909090909,
    "smooth": 0.18039772727272727,
    "left_edge": 0.20933948863636365,
    "right_edge": 0.15873579545454544,
    "apex": 0.19655539772727273
  }
}
```

## Gate A: Tiny Memorization

Gate A asks whether the direct model can fit a fixed synthetic slice at all.

### `w128 d4`

- best epoch: `483`
- val point accuracy: `0.9638`
- val macro recall: `0.9758`
- val pattern accuracy: `0.7891`
- LBFGS accepted: `True`

![Gate A history](../experiment_runs/real_sim/cover_layer_element_branch_synth_mastery_20260314/gate_a_w128_d4/history.png)

### `w256 d6`

- best epoch: `544`
- val point accuracy: `0.9968`
- val macro recall: `0.9978`
- val pattern accuracy: `0.9805`
- LBFGS accepted: `True`

![Gate A history](../experiment_runs/real_sim/cover_layer_element_branch_synth_mastery_20260314/gate_a_w256_d6/history.png)

Gate A **passed**.

- winner: `w256 d6`

## Gate B: Fixed Medium Synthetic Mastery

### `w256 d6`

- val point accuracy: `0.3080`
- val macro recall: `0.3582`
- val pattern accuracy: `0.0068`
![Gate B history](../experiment_runs/real_sim/cover_layer_element_branch_synth_mastery_20260314/gate_b_w256_d6/history.png)

### `w128 d4`

- val point accuracy: `0.3387`
- val macro recall: `0.3532`
- val pattern accuracy: `0.0039`
![Gate B history](../experiment_runs/real_sim/cover_layer_element_branch_synth_mastery_20260314/gate_b_alt_w128_d4/history.png)

Gate B passed: `False`

## Conclusion

This phase is successful only if the direct raw-element MLP masters its **synthetic** domain first.

- If Gate A fails, the issue is no longer just schedule length; there is a deeper model/data-alignment problem.
- If Gate A passes but Gate B or C fail, the issue is likely full-pattern label complexity or model inductive bias.
- Only after synthetic mastery should the real diagnostic gap be used to argue about generator-domain mismatch.
