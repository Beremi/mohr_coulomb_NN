# Cover Layer Branch Predictor Resample Post-Train Report

## Summary

- base checkpoint: `experiment_runs/real_sim/cover_layer_branch_predictor_expert_principal_w1024_d6_acceptall_p1_1000_capped2_20260315/best.pt`
- model: `w1024 d6`
- feature set: `trial_raw_material`
- training dataset per resample: `81920` pointwise samples
- training batch size: `8192` pointwise samples
- optimizer: `lbfgs`
- fixed LR: `1.0e-01`
- per-loop patience before resampling: `1`
- acceptance mode: `improve_only`
- attempted loops: `100`
- accepted loops: `5`

## Baseline vs Final

- baseline synthetic test accuracy / macro recall: `0.9659` / `0.9647`
- final synthetic test accuracy / macro recall: `0.9666` / `0.9654`
- baseline real test accuracy / macro recall: `0.8972` / `0.8878`
- final real test accuracy / macro recall: `0.8972` / `0.8876`

## Loop Results

- loop `1` accepted: `False`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `2` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9662`
  best loop real test: acc `0.8968`, macro `0.8874`
  epochs run before resample/stop: `1`
- loop `3` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9660`
  best loop real test: acc `0.8963`, macro `0.8863`
  epochs run before resample/stop: `1`
- loop `4` accepted: `False`
  best synthetic val score: macro `0.9652`, acc `0.9660`
  best loop real test: acc `0.8963`, macro `0.8863`
  epochs run before resample/stop: `1`
- loop `5` accepted: `False`
  best synthetic val score: macro `0.9652`, acc `0.9660`
  best loop real test: acc `0.8963`, macro `0.8863`
  epochs run before resample/stop: `1`
- loop `6` accepted: `False`
  best synthetic val score: macro `0.9652`, acc `0.9660`
  best loop real test: acc `0.8963`, macro `0.8863`
  epochs run before resample/stop: `1`
- loop `7` accepted: `False`
  best synthetic val score: macro `0.9652`, acc `0.9660`
  best loop real test: acc `0.8963`, macro `0.8863`
  epochs run before resample/stop: `1`
- loop `8` accepted: `False`
  best synthetic val score: macro `0.9652`, acc `0.9660`
  best loop real test: acc `0.8963`, macro `0.8863`
  epochs run before resample/stop: `1`
- loop `9` accepted: `False`
  best synthetic val score: macro `0.9652`, acc `0.9660`
  best loop real test: acc `0.8963`, macro `0.8863`
  epochs run before resample/stop: `1`
- loop `10` accepted: `False`
  best synthetic val score: macro `0.9652`, acc `0.9660`
  best loop real test: acc `0.8963`, macro `0.8863`
  epochs run before resample/stop: `1`
- loop `11` accepted: `False`
  best synthetic val score: macro `0.9652`, acc `0.9660`
  best loop real test: acc `0.8963`, macro `0.8863`
  epochs run before resample/stop: `1`
- loop `12` accepted: `False`
  best synthetic val score: macro `0.9652`, acc `0.9660`
  best loop real test: acc `0.8963`, macro `0.8863`
  epochs run before resample/stop: `1`
- loop `13` accepted: `False`
  best synthetic val score: macro `0.9652`, acc `0.9660`
  best loop real test: acc `0.8963`, macro `0.8863`
  epochs run before resample/stop: `1`
- loop `14` accepted: `False`
  best synthetic val score: macro `0.9652`, acc `0.9660`
  best loop real test: acc `0.8963`, macro `0.8863`
  epochs run before resample/stop: `1`
- loop `15` accepted: `False`
  best synthetic val score: macro `0.9652`, acc `0.9660`
  best loop real test: acc `0.8963`, macro `0.8863`
  epochs run before resample/stop: `1`
- loop `16` accepted: `False`
  best synthetic val score: macro `0.9652`, acc `0.9660`
  best loop real test: acc `0.8963`, macro `0.8863`
  epochs run before resample/stop: `1`
- loop `17` accepted: `True`
  best synthetic val score: macro `0.9655`, acc `0.9662`
  best loop real test: acc `0.9007`, macro `0.8916`
  epochs run before resample/stop: `1`
- loop `18` accepted: `False`
  best synthetic val score: macro `0.9655`, acc `0.9662`
  best loop real test: acc `0.9007`, macro `0.8916`
  epochs run before resample/stop: `1`
- loop `19` accepted: `False`
  best synthetic val score: macro `0.9655`, acc `0.9662`
  best loop real test: acc `0.9007`, macro `0.8916`
  epochs run before resample/stop: `1`
- loop `20` accepted: `False`
  best synthetic val score: macro `0.9655`, acc `0.9662`
  best loop real test: acc `0.9007`, macro `0.8916`
  epochs run before resample/stop: `1`
- loop `21` accepted: `False`
  best synthetic val score: macro `0.9655`, acc `0.9662`
  best loop real test: acc `0.9007`, macro `0.8916`
  epochs run before resample/stop: `1`
- loop `22` accepted: `False`
  best synthetic val score: macro `0.9655`, acc `0.9662`
  best loop real test: acc `0.9007`, macro `0.8916`
  epochs run before resample/stop: `1`
- loop `23` accepted: `False`
  best synthetic val score: macro `0.9655`, acc `0.9662`
  best loop real test: acc `0.9007`, macro `0.8916`
  epochs run before resample/stop: `1`
- loop `24` accepted: `False`
  best synthetic val score: macro `0.9655`, acc `0.9662`
  best loop real test: acc `0.9007`, macro `0.8916`
  epochs run before resample/stop: `1`
- loop `25` accepted: `False`
  best synthetic val score: macro `0.9655`, acc `0.9662`
  best loop real test: acc `0.9007`, macro `0.8916`
  epochs run before resample/stop: `1`
- loop `26` accepted: `False`
  best synthetic val score: macro `0.9655`, acc `0.9662`
  best loop real test: acc `0.9007`, macro `0.8916`
  epochs run before resample/stop: `1`
- loop `27` accepted: `False`
  best synthetic val score: macro `0.9655`, acc `0.9662`
  best loop real test: acc `0.9007`, macro `0.8916`
  epochs run before resample/stop: `1`
- loop `28` accepted: `True`
  best synthetic val score: macro `0.9655`, acc `0.9664`
  best loop real test: acc `0.8924`, macro `0.8826`
  epochs run before resample/stop: `1`
- loop `29` accepted: `True`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `30` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `31` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `32` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `33` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `34` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `35` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `36` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `37` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `38` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `39` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `40` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `41` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `42` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `43` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `44` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `45` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `46` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `47` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `48` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `49` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `50` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `51` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `52` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `53` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `54` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `55` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `56` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `57` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `58` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `59` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `60` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `61` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `62` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `63` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `64` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `65` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `66` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `67` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `68` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `69` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `70` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `71` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `72` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `73` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `74` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `75` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `76` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `77` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `78` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `79` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `80` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `81` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `82` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `83` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `84` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `85` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `86` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `87` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `88` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `89` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `90` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `91` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `92` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `93` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `94` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `95` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `96` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `97` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `98` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `99` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `100` accepted: `False`
  best synthetic val score: macro `0.9658`, acc `0.9667`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`

## Curves

![Training history](../experiment_runs/real_sim/cover_layer_branch_predictor_expert_principal_w1024_d6_lbfgs_100cycles_20260315/training_history.png)
