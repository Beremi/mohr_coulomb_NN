# Cover Layer Branch Predictor Resample Post-Train Report

## Summary

- base checkpoint: `experiment_runs/real_sim/cover_layer_branch_predictor_expert_principal_w1024_d6_acceptall_p1_1000_capped2_20260315/best.pt`
- model: `w1024 d6`
- feature set: `trial_raw_material`
- training dataset per resample: `81920` pointwise samples
- training batch size: `8192` pointwise samples
- replay mode: `misprediction_bank`
- replay cap per cycle: `16384`
- final replay bank size: `678015`
- optimizer: `lbfgs`
- fixed LR: `1.0e-01`
- per-loop patience before resampling: `1`
- eval/checkpoint interval: every `10` loops
- acceptance mode: `improve_only`
- attempted loops: `1000`
- accepted loops: `4`

## Baseline vs Final

- baseline synthetic test accuracy / macro recall: `0.9659` / `0.9647`
- final synthetic test accuracy / macro recall: `0.9663` / `0.9652`
- baseline real test accuracy / macro recall: `0.8972` / `0.8878`
- final real test accuracy / macro recall: `0.8929` / `0.8832`

## Important Nuance

- accepted checkpoints were written at loops `540`, `650`, `660`, and `900`
- the final accepted endpoint is the best checkpoint under the fixed synthetic-score acceptance rule
- it is **not** the best real-data point seen during the 1000-loop walk
- best real test accuracy observed in the trajectory:
  - loop `300`: accuracy `0.9009`, macro recall `0.8916`
- best real test macro recall observed in the trajectory:
  - loop `280`: accuracy `0.9006`, macro recall `0.8918`
- those better real points were not accepted because their synthetic validation score did not beat the currently accepted synthetic-best checkpoint
- practical interpretation:
  - replay-bank LBFGS did find stronger real pockets
  - but the synthetic acceptance objective still diverged from the real objective

## Loop Results

- loop `10` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `20` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `30` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `40` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `50` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `60` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `70` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `80` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `90` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `100` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `110` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `120` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `130` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `140` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `150` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `160` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `170` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `180` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `190` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `200` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `210` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `220` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `230` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `240` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `250` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `260` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `270` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `280` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `290` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `300` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `310` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `320` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `330` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `340` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `350` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `360` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `370` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `380` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `390` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `400` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `410` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `420` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `430` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `440` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `450` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `460` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `470` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `480` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `490` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `500` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `510` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `520` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `530` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `540` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9657`
  best loop real test: acc `0.8947`, macro `0.8848`
  epochs run before resample/stop: `1`
- loop `550` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9657`
  best loop real test: acc `0.8947`, macro `0.8848`
  epochs run before resample/stop: `1`
- loop `560` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9657`
  best loop real test: acc `0.8947`, macro `0.8848`
  epochs run before resample/stop: `1`
- loop `570` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9657`
  best loop real test: acc `0.8947`, macro `0.8848`
  epochs run before resample/stop: `1`
- loop `580` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9657`
  best loop real test: acc `0.8947`, macro `0.8848`
  epochs run before resample/stop: `1`
- loop `590` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9657`
  best loop real test: acc `0.8947`, macro `0.8848`
  epochs run before resample/stop: `1`
- loop `600` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9657`
  best loop real test: acc `0.8947`, macro `0.8848`
  epochs run before resample/stop: `1`
- loop `610` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9657`
  best loop real test: acc `0.8947`, macro `0.8848`
  epochs run before resample/stop: `1`
- loop `620` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9657`
  best loop real test: acc `0.8947`, macro `0.8848`
  epochs run before resample/stop: `1`
- loop `630` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9657`
  best loop real test: acc `0.8947`, macro `0.8848`
  epochs run before resample/stop: `1`
- loop `640` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9657`
  best loop real test: acc `0.8947`, macro `0.8848`
  epochs run before resample/stop: `1`
- loop `650` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9663`
  best loop real test: acc `0.8956`, macro `0.8860`
  epochs run before resample/stop: `1`
- loop `660` accepted: `True`
  best synthetic val score: macro `0.9654`, acc `0.9666`
  best loop real test: acc `0.8961`, macro `0.8865`
  epochs run before resample/stop: `1`
- loop `670` accepted: `False`
  best synthetic val score: macro `0.9654`, acc `0.9666`
  best loop real test: acc `0.8961`, macro `0.8865`
  epochs run before resample/stop: `1`
- loop `680` accepted: `False`
  best synthetic val score: macro `0.9654`, acc `0.9666`
  best loop real test: acc `0.8961`, macro `0.8865`
  epochs run before resample/stop: `1`
- loop `690` accepted: `False`
  best synthetic val score: macro `0.9654`, acc `0.9666`
  best loop real test: acc `0.8961`, macro `0.8865`
  epochs run before resample/stop: `1`
- loop `700` accepted: `False`
  best synthetic val score: macro `0.9654`, acc `0.9666`
  best loop real test: acc `0.8961`, macro `0.8865`
  epochs run before resample/stop: `1`
- loop `710` accepted: `False`
  best synthetic val score: macro `0.9654`, acc `0.9666`
  best loop real test: acc `0.8961`, macro `0.8865`
  epochs run before resample/stop: `1`
- loop `720` accepted: `False`
  best synthetic val score: macro `0.9654`, acc `0.9666`
  best loop real test: acc `0.8961`, macro `0.8865`
  epochs run before resample/stop: `1`
- loop `730` accepted: `False`
  best synthetic val score: macro `0.9654`, acc `0.9666`
  best loop real test: acc `0.8961`, macro `0.8865`
  epochs run before resample/stop: `1`
- loop `740` accepted: `False`
  best synthetic val score: macro `0.9654`, acc `0.9666`
  best loop real test: acc `0.8961`, macro `0.8865`
  epochs run before resample/stop: `1`
- loop `750` accepted: `False`
  best synthetic val score: macro `0.9654`, acc `0.9666`
  best loop real test: acc `0.8961`, macro `0.8865`
  epochs run before resample/stop: `1`
- loop `760` accepted: `False`
  best synthetic val score: macro `0.9654`, acc `0.9666`
  best loop real test: acc `0.8961`, macro `0.8865`
  epochs run before resample/stop: `1`
- loop `770` accepted: `False`
  best synthetic val score: macro `0.9654`, acc `0.9666`
  best loop real test: acc `0.8961`, macro `0.8865`
  epochs run before resample/stop: `1`
- loop `780` accepted: `False`
  best synthetic val score: macro `0.9654`, acc `0.9666`
  best loop real test: acc `0.8961`, macro `0.8865`
  epochs run before resample/stop: `1`
- loop `790` accepted: `False`
  best synthetic val score: macro `0.9654`, acc `0.9666`
  best loop real test: acc `0.8961`, macro `0.8865`
  epochs run before resample/stop: `1`
- loop `800` accepted: `False`
  best synthetic val score: macro `0.9654`, acc `0.9666`
  best loop real test: acc `0.8961`, macro `0.8865`
  epochs run before resample/stop: `1`
- loop `810` accepted: `False`
  best synthetic val score: macro `0.9654`, acc `0.9666`
  best loop real test: acc `0.8961`, macro `0.8865`
  epochs run before resample/stop: `1`
- loop `820` accepted: `False`
  best synthetic val score: macro `0.9654`, acc `0.9666`
  best loop real test: acc `0.8961`, macro `0.8865`
  epochs run before resample/stop: `1`
- loop `830` accepted: `False`
  best synthetic val score: macro `0.9654`, acc `0.9666`
  best loop real test: acc `0.8961`, macro `0.8865`
  epochs run before resample/stop: `1`
- loop `840` accepted: `False`
  best synthetic val score: macro `0.9654`, acc `0.9666`
  best loop real test: acc `0.8961`, macro `0.8865`
  epochs run before resample/stop: `1`
- loop `850` accepted: `False`
  best synthetic val score: macro `0.9654`, acc `0.9666`
  best loop real test: acc `0.8961`, macro `0.8865`
  epochs run before resample/stop: `1`
- loop `860` accepted: `False`
  best synthetic val score: macro `0.9654`, acc `0.9666`
  best loop real test: acc `0.8961`, macro `0.8865`
  epochs run before resample/stop: `1`
- loop `870` accepted: `False`
  best synthetic val score: macro `0.9654`, acc `0.9666`
  best loop real test: acc `0.8961`, macro `0.8865`
  epochs run before resample/stop: `1`
- loop `880` accepted: `False`
  best synthetic val score: macro `0.9654`, acc `0.9666`
  best loop real test: acc `0.8961`, macro `0.8865`
  epochs run before resample/stop: `1`
- loop `890` accepted: `False`
  best synthetic val score: macro `0.9654`, acc `0.9666`
  best loop real test: acc `0.8961`, macro `0.8865`
  epochs run before resample/stop: `1`
- loop `900` accepted: `True`
  best synthetic val score: macro `0.9655`, acc `0.9663`
  best loop real test: acc `0.8929`, macro `0.8832`
  epochs run before resample/stop: `1`
- loop `910` accepted: `False`
  best synthetic val score: macro `0.9655`, acc `0.9663`
  best loop real test: acc `0.8929`, macro `0.8832`
  epochs run before resample/stop: `1`
- loop `920` accepted: `False`
  best synthetic val score: macro `0.9655`, acc `0.9663`
  best loop real test: acc `0.8929`, macro `0.8832`
  epochs run before resample/stop: `1`
- loop `930` accepted: `False`
  best synthetic val score: macro `0.9655`, acc `0.9663`
  best loop real test: acc `0.8929`, macro `0.8832`
  epochs run before resample/stop: `1`
- loop `940` accepted: `False`
  best synthetic val score: macro `0.9655`, acc `0.9663`
  best loop real test: acc `0.8929`, macro `0.8832`
  epochs run before resample/stop: `1`
- loop `950` accepted: `False`
  best synthetic val score: macro `0.9655`, acc `0.9663`
  best loop real test: acc `0.8929`, macro `0.8832`
  epochs run before resample/stop: `1`
- loop `960` accepted: `False`
  best synthetic val score: macro `0.9655`, acc `0.9663`
  best loop real test: acc `0.8929`, macro `0.8832`
  epochs run before resample/stop: `1`
- loop `970` accepted: `False`
  best synthetic val score: macro `0.9655`, acc `0.9663`
  best loop real test: acc `0.8929`, macro `0.8832`
  epochs run before resample/stop: `1`
- loop `980` accepted: `False`
  best synthetic val score: macro `0.9655`, acc `0.9663`
  best loop real test: acc `0.8929`, macro `0.8832`
  epochs run before resample/stop: `1`
- loop `990` accepted: `False`
  best synthetic val score: macro `0.9655`, acc `0.9663`
  best loop real test: acc `0.8929`, macro `0.8832`
  epochs run before resample/stop: `1`
- loop `1000` accepted: `False`
  best synthetic val score: macro `0.9655`, acc `0.9663`
  best loop real test: acc `0.8929`, macro `0.8832`
  epochs run before resample/stop: `1`

## Curves

![Training history](../experiment_runs/real_sim/cover_layer_branch_predictor_expert_principal_w1024_d6_lbfgs_1000cycles_replay_eval10_cap16384_20260315/training_history.png)
