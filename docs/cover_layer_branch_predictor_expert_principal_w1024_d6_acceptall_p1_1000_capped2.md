# Cover Layer Branch Predictor Resample Post-Train Report

## Summary

- base checkpoint: `experiment_runs/real_sim/cover_layer_branch_predictor_expert_principal_w1024_d6_resample_10xdata_lr1e6_20260315/best.pt`
- model: `w1024 d6`
- feature set: `trial_raw_material`
- training dataset per resample: `81920` pointwise samples
- training batch size: `8192` pointwise samples
- fixed LR: `1.0e-06`
- per-loop patience before resampling: `1`
- acceptance mode: `accept_all_loops`
- attempted loops: `1000`
- accepted loops: `1000`

## Baseline vs Final

- baseline synthetic test accuracy / macro recall: `0.9663` / `0.9656`
- final synthetic test accuracy / macro recall: `0.9659` / `0.9647`
- baseline real test accuracy / macro recall: `0.8956` / `0.8859`
- final real test accuracy / macro recall: `0.8972` / `0.8878`

## Loop Results

- loop `1` accepted: `True`
  best synthetic val score: macro `0.9662`, acc `0.9666`
  best loop real test: acc `0.8961`, macro `0.8865`
  epochs run before resample/stop: `1`
- loop `2` accepted: `True`
  best synthetic val score: macro `0.9661`, acc `0.9665`
  best loop real test: acc `0.8960`, macro `0.8863`
  epochs run before resample/stop: `1`
- loop `3` accepted: `True`
  best synthetic val score: macro `0.9661`, acc `0.9665`
  best loop real test: acc `0.8956`, macro `0.8859`
  epochs run before resample/stop: `2`
- loop `4` accepted: `True`
  best synthetic val score: macro `0.9661`, acc `0.9665`
  best loop real test: acc `0.8952`, macro `0.8855`
  epochs run before resample/stop: `1`
- loop `5` accepted: `True`
  best synthetic val score: macro `0.9661`, acc `0.9665`
  best loop real test: acc `0.8958`, macro `0.8861`
  epochs run before resample/stop: `1`
- loop `6` accepted: `True`
  best synthetic val score: macro `0.9660`, acc `0.9664`
  best loop real test: acc `0.8963`, macro `0.8867`
  epochs run before resample/stop: `1`
- loop `7` accepted: `True`
  best synthetic val score: macro `0.9660`, acc `0.9664`
  best loop real test: acc `0.8961`, macro `0.8864`
  epochs run before resample/stop: `1`
- loop `8` accepted: `True`
  best synthetic val score: macro `0.9660`, acc `0.9664`
  best loop real test: acc `0.8961`, macro `0.8865`
  epochs run before resample/stop: `2`
- loop `9` accepted: `True`
  best synthetic val score: macro `0.9660`, acc `0.9664`
  best loop real test: acc `0.8961`, macro `0.8865`
  epochs run before resample/stop: `2`
- loop `10` accepted: `True`
  best synthetic val score: macro `0.9659`, acc `0.9664`
  best loop real test: acc `0.8961`, macro `0.8865`
  epochs run before resample/stop: `1`
- loop `11` accepted: `True`
  best synthetic val score: macro `0.9660`, acc `0.9665`
  best loop real test: acc `0.8961`, macro `0.8863`
  epochs run before resample/stop: `2`
- loop `12` accepted: `True`
  best synthetic val score: macro `0.9660`, acc `0.9665`
  best loop real test: acc `0.8958`, macro `0.8859`
  epochs run before resample/stop: `1`
- loop `13` accepted: `True`
  best synthetic val score: macro `0.9660`, acc `0.9665`
  best loop real test: acc `0.8952`, macro `0.8854`
  epochs run before resample/stop: `1`
- loop `14` accepted: `True`
  best synthetic val score: macro `0.9659`, acc `0.9664`
  best loop real test: acc `0.8956`, macro `0.8857`
  epochs run before resample/stop: `1`
- loop `15` accepted: `True`
  best synthetic val score: macro `0.9660`, acc `0.9665`
  best loop real test: acc `0.8961`, macro `0.8864`
  epochs run before resample/stop: `2`
- loop `16` accepted: `True`
  best synthetic val score: macro `0.9660`, acc `0.9665`
  best loop real test: acc `0.8960`, macro `0.8862`
  epochs run before resample/stop: `1`
- loop `17` accepted: `True`
  best synthetic val score: macro `0.9660`, acc `0.9665`
  best loop real test: acc `0.8956`, macro `0.8858`
  epochs run before resample/stop: `2`
- loop `18` accepted: `True`
  best synthetic val score: macro `0.9660`, acc `0.9665`
  best loop real test: acc `0.8960`, macro `0.8862`
  epochs run before resample/stop: `1`
- loop `19` accepted: `True`
  best synthetic val score: macro `0.9660`, acc `0.9665`
  best loop real test: acc `0.8960`, macro `0.8862`
  epochs run before resample/stop: `1`
- loop `20` accepted: `True`
  best synthetic val score: macro `0.9659`, acc `0.9664`
  best loop real test: acc `0.8965`, macro `0.8868`
  epochs run before resample/stop: `2`
- loop `21` accepted: `True`
  best synthetic val score: macro `0.9659`, acc `0.9664`
  best loop real test: acc `0.8956`, macro `0.8858`
  epochs run before resample/stop: `2`
- loop `22` accepted: `True`
  best synthetic val score: macro `0.9660`, acc `0.9665`
  best loop real test: acc `0.8963`, macro `0.8866`
  epochs run before resample/stop: `2`
- loop `23` accepted: `True`
  best synthetic val score: macro `0.9660`, acc `0.9665`
  best loop real test: acc `0.8963`, macro `0.8866`
  epochs run before resample/stop: `2`
- loop `24` accepted: `True`
  best synthetic val score: macro `0.9659`, acc `0.9665`
  best loop real test: acc `0.8967`, macro `0.8870`
  epochs run before resample/stop: `1`
- loop `25` accepted: `True`
  best synthetic val score: macro `0.9659`, acc `0.9664`
  best loop real test: acc `0.8968`, macro `0.8872`
  epochs run before resample/stop: `1`
- loop `26` accepted: `True`
  best synthetic val score: macro `0.9659`, acc `0.9665`
  best loop real test: acc `0.8965`, macro `0.8868`
  epochs run before resample/stop: `2`
- loop `27` accepted: `True`
  best synthetic val score: macro `0.9659`, acc `0.9665`
  best loop real test: acc `0.8970`, macro `0.8874`
  epochs run before resample/stop: `1`
- loop `28` accepted: `True`
  best synthetic val score: macro `0.9657`, acc `0.9664`
  best loop real test: acc `0.8963`, macro `0.8866`
  epochs run before resample/stop: `1`
- loop `29` accepted: `True`
  best synthetic val score: macro `0.9659`, acc `0.9665`
  best loop real test: acc `0.8970`, macro `0.8874`
  epochs run before resample/stop: `2`
- loop `30` accepted: `True`
  best synthetic val score: macro `0.9659`, acc `0.9664`
  best loop real test: acc `0.8967`, macro `0.8869`
  epochs run before resample/stop: `1`
- loop `31` accepted: `True`
  best synthetic val score: macro `0.9659`, acc `0.9665`
  best loop real test: acc `0.8970`, macro `0.8874`
  epochs run before resample/stop: `2`
- loop `32` accepted: `True`
  best synthetic val score: macro `0.9659`, acc `0.9665`
  best loop real test: acc `0.8961`, macro `0.8865`
  epochs run before resample/stop: `2`
- loop `33` accepted: `True`
  best synthetic val score: macro `0.9659`, acc `0.9664`
  best loop real test: acc `0.8965`, macro `0.8869`
  epochs run before resample/stop: `1`
- loop `34` accepted: `True`
  best synthetic val score: macro `0.9658`, acc `0.9664`
  best loop real test: acc `0.8967`, macro `0.8870`
  epochs run before resample/stop: `1`
- loop `35` accepted: `True`
  best synthetic val score: macro `0.9657`, acc `0.9663`
  best loop real test: acc `0.8967`, macro `0.8870`
  epochs run before resample/stop: `1`
- loop `36` accepted: `True`
  best synthetic val score: macro `0.9656`, acc `0.9663`
  best loop real test: acc `0.8967`, macro `0.8870`
  epochs run before resample/stop: `1`
- loop `37` accepted: `True`
  best synthetic val score: macro `0.9656`, acc `0.9663`
  best loop real test: acc `0.8968`, macro `0.8873`
  epochs run before resample/stop: `1`
- loop `38` accepted: `True`
  best synthetic val score: macro `0.9656`, acc `0.9663`
  best loop real test: acc `0.8967`, macro `0.8870`
  epochs run before resample/stop: `2`
- loop `39` accepted: `True`
  best synthetic val score: macro `0.9656`, acc `0.9663`
  best loop real test: acc `0.8968`, macro `0.8872`
  epochs run before resample/stop: `1`
- loop `40` accepted: `True`
  best synthetic val score: macro `0.9656`, acc `0.9662`
  best loop real test: acc `0.8970`, macro `0.8874`
  epochs run before resample/stop: `1`
- loop `41` accepted: `True`
  best synthetic val score: macro `0.9655`, acc `0.9662`
  best loop real test: acc `0.8970`, macro `0.8874`
  epochs run before resample/stop: `1`
- loop `42` accepted: `True`
  best synthetic val score: macro `0.9656`, acc `0.9662`
  best loop real test: acc `0.8967`, macro `0.8871`
  epochs run before resample/stop: `2`
- loop `43` accepted: `True`
  best synthetic val score: macro `0.9656`, acc `0.9662`
  best loop real test: acc `0.8965`, macro `0.8868`
  epochs run before resample/stop: `1`
- loop `44` accepted: `True`
  best synthetic val score: macro `0.9656`, acc `0.9662`
  best loop real test: acc `0.8967`, macro `0.8870`
  epochs run before resample/stop: `2`
- loop `45` accepted: `True`
  best synthetic val score: macro `0.9655`, acc `0.9662`
  best loop real test: acc `0.8968`, macro `0.8872`
  epochs run before resample/stop: `1`
- loop `46` accepted: `True`
  best synthetic val score: macro `0.9655`, acc `0.9662`
  best loop real test: acc `0.8965`, macro `0.8868`
  epochs run before resample/stop: `2`
- loop `47` accepted: `True`
  best synthetic val score: macro `0.9655`, acc `0.9662`
  best loop real test: acc `0.8965`, macro `0.8869`
  epochs run before resample/stop: `1`
- loop `48` accepted: `True`
  best synthetic val score: macro `0.9655`, acc `0.9662`
  best loop real test: acc `0.8965`, macro `0.8868`
  epochs run before resample/stop: `1`
- loop `49` accepted: `True`
  best synthetic val score: macro `0.9655`, acc `0.9662`
  best loop real test: acc `0.8970`, macro `0.8874`
  epochs run before resample/stop: `2`
- loop `50` accepted: `True`
  best synthetic val score: macro `0.9655`, acc `0.9662`
  best loop real test: acc `0.8970`, macro `0.8874`
  epochs run before resample/stop: `2`
- loop `51` accepted: `True`
  best synthetic val score: macro `0.9655`, acc `0.9662`
  best loop real test: acc `0.8965`, macro `0.8868`
  epochs run before resample/stop: `2`
- loop `52` accepted: `True`
  best synthetic val score: macro `0.9655`, acc `0.9662`
  best loop real test: acc `0.8970`, macro `0.8874`
  epochs run before resample/stop: `1`
- loop `53` accepted: `True`
  best synthetic val score: macro `0.9655`, acc `0.9662`
  best loop real test: acc `0.8967`, macro `0.8871`
  epochs run before resample/stop: `1`
- loop `54` accepted: `True`
  best synthetic val score: macro `0.9655`, acc `0.9661`
  best loop real test: acc `0.8967`, macro `0.8871`
  epochs run before resample/stop: `1`
- loop `55` accepted: `True`
  best synthetic val score: macro `0.9655`, acc `0.9661`
  best loop real test: acc `0.8965`, macro `0.8869`
  epochs run before resample/stop: `1`
- loop `56` accepted: `True`
  best synthetic val score: macro `0.9655`, acc `0.9662`
  best loop real test: acc `0.8968`, macro `0.8872`
  epochs run before resample/stop: `2`
- loop `57` accepted: `True`
  best synthetic val score: macro `0.9655`, acc `0.9662`
  best loop real test: acc `0.8970`, macro `0.8874`
  epochs run before resample/stop: `2`
- loop `58` accepted: `True`
  best synthetic val score: macro `0.9655`, acc `0.9662`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `59` accepted: `True`
  best synthetic val score: macro `0.9654`, acc `0.9661`
  best loop real test: acc `0.8970`, macro `0.8874`
  epochs run before resample/stop: `2`
- loop `60` accepted: `True`
  best synthetic val score: macro `0.9654`, acc `0.9661`
  best loop real test: acc `0.8974`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `61` accepted: `True`
  best synthetic val score: macro `0.9655`, acc `0.9662`
  best loop real test: acc `0.8970`, macro `0.8874`
  epochs run before resample/stop: `2`
- loop `62` accepted: `True`
  best synthetic val score: macro `0.9655`, acc `0.9662`
  best loop real test: acc `0.8970`, macro `0.8874`
  epochs run before resample/stop: `1`
- loop `63` accepted: `True`
  best synthetic val score: macro `0.9655`, acc `0.9662`
  best loop real test: acc `0.8968`, macro `0.8872`
  epochs run before resample/stop: `1`
- loop `64` accepted: `True`
  best synthetic val score: macro `0.9655`, acc `0.9662`
  best loop real test: acc `0.8970`, macro `0.8874`
  epochs run before resample/stop: `1`
- loop `65` accepted: `True`
  best synthetic val score: macro `0.9655`, acc `0.9662`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `66` accepted: `True`
  best synthetic val score: macro `0.9654`, acc `0.9662`
  best loop real test: acc `0.8970`, macro `0.8874`
  epochs run before resample/stop: `1`
- loop `67` accepted: `True`
  best synthetic val score: macro `0.9654`, acc `0.9661`
  best loop real test: acc `0.8968`, macro `0.8873`
  epochs run before resample/stop: `2`
- loop `68` accepted: `True`
  best synthetic val score: macro `0.9654`, acc `0.9662`
  best loop real test: acc `0.8968`, macro `0.8872`
  epochs run before resample/stop: `1`
- loop `69` accepted: `True`
  best synthetic val score: macro `0.9654`, acc `0.9661`
  best loop real test: acc `0.8968`, macro `0.8872`
  epochs run before resample/stop: `1`
- loop `70` accepted: `True`
  best synthetic val score: macro `0.9654`, acc `0.9661`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `71` accepted: `True`
  best synthetic val score: macro `0.9654`, acc `0.9662`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `2`
- loop `72` accepted: `True`
  best synthetic val score: macro `0.9654`, acc `0.9662`
  best loop real test: acc `0.8970`, macro `0.8874`
  epochs run before resample/stop: `1`
- loop `73` accepted: `True`
  best synthetic val score: macro `0.9654`, acc `0.9662`
  best loop real test: acc `0.8972`, macro `0.8875`
  epochs run before resample/stop: `1`
- loop `74` accepted: `True`
  best synthetic val score: macro `0.9653`, acc `0.9662`
  best loop real test: acc `0.8968`, macro `0.8871`
  epochs run before resample/stop: `1`
- loop `75` accepted: `True`
  best synthetic val score: macro `0.9653`, acc `0.9661`
  best loop real test: acc `0.8970`, macro `0.8874`
  epochs run before resample/stop: `2`
- loop `76` accepted: `True`
  best synthetic val score: macro `0.9653`, acc `0.9661`
  best loop real test: acc `0.8970`, macro `0.8874`
  epochs run before resample/stop: `1`
- loop `77` accepted: `True`
  best synthetic val score: macro `0.9654`, acc `0.9661`
  best loop real test: acc `0.8968`, macro `0.8872`
  epochs run before resample/stop: `2`
- loop `78` accepted: `True`
  best synthetic val score: macro `0.9654`, acc `0.9661`
  best loop real test: acc `0.8968`, macro `0.8872`
  epochs run before resample/stop: `2`
- loop `79` accepted: `True`
  best synthetic val score: macro `0.9653`, acc `0.9661`
  best loop real test: acc `0.8968`, macro `0.8872`
  epochs run before resample/stop: `1`
- loop `80` accepted: `True`
  best synthetic val score: macro `0.9653`, acc `0.9661`
  best loop real test: acc `0.8968`, macro `0.8873`
  epochs run before resample/stop: `2`
- loop `81` accepted: `True`
  best synthetic val score: macro `0.9653`, acc `0.9661`
  best loop real test: acc `0.8965`, macro `0.8869`
  epochs run before resample/stop: `1`
- loop `82` accepted: `True`
  best synthetic val score: macro `0.9654`, acc `0.9662`
  best loop real test: acc `0.8967`, macro `0.8871`
  epochs run before resample/stop: `2`
- loop `83` accepted: `True`
  best synthetic val score: macro `0.9654`, acc `0.9662`
  best loop real test: acc `0.8965`, macro `0.8869`
  epochs run before resample/stop: `2`
- loop `84` accepted: `True`
  best synthetic val score: macro `0.9654`, acc `0.9662`
  best loop real test: acc `0.8967`, macro `0.8871`
  epochs run before resample/stop: `2`
- loop `85` accepted: `True`
  best synthetic val score: macro `0.9654`, acc `0.9662`
  best loop real test: acc `0.8968`, macro `0.8872`
  epochs run before resample/stop: `2`
- loop `86` accepted: `True`
  best synthetic val score: macro `0.9654`, acc `0.9662`
  best loop real test: acc `0.8968`, macro `0.8872`
  epochs run before resample/stop: `1`
- loop `87` accepted: `True`
  best synthetic val score: macro `0.9654`, acc `0.9662`
  best loop real test: acc `0.8968`, macro `0.8872`
  epochs run before resample/stop: `1`
- loop `88` accepted: `True`
  best synthetic val score: macro `0.9653`, acc `0.9661`
  best loop real test: acc `0.8968`, macro `0.8872`
  epochs run before resample/stop: `1`
- loop `89` accepted: `True`
  best synthetic val score: macro `0.9653`, acc `0.9661`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `2`
- loop `90` accepted: `True`
  best synthetic val score: macro `0.9654`, acc `0.9662`
  best loop real test: acc `0.8974`, macro `0.8878`
  epochs run before resample/stop: `2`
- loop `91` accepted: `True`
  best synthetic val score: macro `0.9654`, acc `0.9661`
  best loop real test: acc `0.8977`, macro `0.8881`
  epochs run before resample/stop: `1`
- loop `92` accepted: `True`
  best synthetic val score: macro `0.9654`, acc `0.9661`
  best loop real test: acc `0.8975`, macro `0.8879`
  epochs run before resample/stop: `2`
- loop `93` accepted: `True`
  best synthetic val score: macro `0.9653`, acc `0.9661`
  best loop real test: acc `0.8977`, macro `0.8881`
  epochs run before resample/stop: `1`
- loop `94` accepted: `True`
  best synthetic val score: macro `0.9653`, acc `0.9661`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `2`
- loop `95` accepted: `True`
  best synthetic val score: macro `0.9654`, acc `0.9662`
  best loop real test: acc `0.8968`, macro `0.8872`
  epochs run before resample/stop: `2`
- loop `96` accepted: `True`
  best synthetic val score: macro `0.9653`, acc `0.9661`
  best loop real test: acc `0.8974`, macro `0.8878`
  epochs run before resample/stop: `2`
- loop `97` accepted: `True`
  best synthetic val score: macro `0.9654`, acc `0.9661`
  best loop real test: acc `0.8975`, macro `0.8880`
  epochs run before resample/stop: `2`
- loop `98` accepted: `True`
  best synthetic val score: macro `0.9654`, acc `0.9661`
  best loop real test: acc `0.8974`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `99` accepted: `True`
  best synthetic val score: macro `0.9653`, acc `0.9661`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `100` accepted: `True`
  best synthetic val score: macro `0.9653`, acc `0.9661`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `101` accepted: `True`
  best synthetic val score: macro `0.9653`, acc `0.9661`
  best loop real test: acc `0.8968`, macro `0.8872`
  epochs run before resample/stop: `1`
- loop `102` accepted: `True`
  best synthetic val score: macro `0.9653`, acc `0.9661`
  best loop real test: acc `0.8967`, macro `0.8870`
  epochs run before resample/stop: `1`
- loop `103` accepted: `True`
  best synthetic val score: macro `0.9653`, acc `0.9661`
  best loop real test: acc `0.8968`, macro `0.8872`
  epochs run before resample/stop: `1`
- loop `104` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9661`
  best loop real test: acc `0.8970`, macro `0.8874`
  epochs run before resample/stop: `1`
- loop `105` accepted: `True`
  best synthetic val score: macro `0.9653`, acc `0.9661`
  best loop real test: acc `0.8968`, macro `0.8872`
  epochs run before resample/stop: `2`
- loop `106` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9660`
  best loop real test: acc `0.8968`, macro `0.8872`
  epochs run before resample/stop: `1`
- loop `107` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9659`
  best loop real test: acc `0.8965`, macro `0.8869`
  epochs run before resample/stop: `1`
- loop `108` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8965`, macro `0.8869`
  epochs run before resample/stop: `1`
- loop `109` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8963`, macro `0.8866`
  epochs run before resample/stop: `1`
- loop `110` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8963`, macro `0.8867`
  epochs run before resample/stop: `1`
- loop `111` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8965`, macro `0.8868`
  epochs run before resample/stop: `1`
- loop `112` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9660`
  best loop real test: acc `0.8970`, macro `0.8874`
  epochs run before resample/stop: `2`
- loop `113` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9661`
  best loop real test: acc `0.8967`, macro `0.8870`
  epochs run before resample/stop: `2`
- loop `114` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9660`
  best loop real test: acc `0.8965`, macro `0.8869`
  epochs run before resample/stop: `1`
- loop `115` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9660`
  best loop real test: acc `0.8968`, macro `0.8873`
  epochs run before resample/stop: `2`
- loop `116` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9660`
  best loop real test: acc `0.8965`, macro `0.8868`
  epochs run before resample/stop: `1`
- loop `117` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8965`, macro `0.8869`
  epochs run before resample/stop: `1`
- loop `118` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8967`, macro `0.8870`
  epochs run before resample/stop: `1`
- loop `119` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8965`, macro `0.8868`
  epochs run before resample/stop: `2`
- loop `120` accepted: `True`
  best synthetic val score: macro `0.9653`, acc `0.9661`
  best loop real test: acc `0.8968`, macro `0.8872`
  epochs run before resample/stop: `2`
- loop `121` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9661`
  best loop real test: acc `0.8970`, macro `0.8874`
  epochs run before resample/stop: `1`
- loop `122` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9660`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `1`
- loop `123` accepted: `True`
  best synthetic val score: macro `0.9653`, acc `0.9661`
  best loop real test: acc `0.8967`, macro `0.8870`
  epochs run before resample/stop: `2`
- loop `124` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9660`
  best loop real test: acc `0.8967`, macro `0.8870`
  epochs run before resample/stop: `1`
- loop `125` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8965`, macro `0.8868`
  epochs run before resample/stop: `1`
- loop `126` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8965`, macro `0.8869`
  epochs run before resample/stop: `1`
- loop `127` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8960`, macro `0.8863`
  epochs run before resample/stop: `1`
- loop `128` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8963`, macro `0.8867`
  epochs run before resample/stop: `2`
- loop `129` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8963`, macro `0.8867`
  epochs run before resample/stop: `2`
- loop `130` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8963`, macro `0.8866`
  epochs run before resample/stop: `1`
- loop `131` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9659`
  best loop real test: acc `0.8967`, macro `0.8871`
  epochs run before resample/stop: `2`
- loop `132` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9659`
  best loop real test: acc `0.8968`, macro `0.8873`
  epochs run before resample/stop: `1`
- loop `133` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `1`
- loop `134` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9659`
  best loop real test: acc `0.8968`, macro `0.8873`
  epochs run before resample/stop: `2`
- loop `135` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9659`
  best loop real test: acc `0.8965`, macro `0.8869`
  epochs run before resample/stop: `2`
- loop `136` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8967`, macro `0.8871`
  epochs run before resample/stop: `1`
- loop `137` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9659`
  best loop real test: acc `0.8967`, macro `0.8871`
  epochs run before resample/stop: `2`
- loop `138` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9659`
  best loop real test: acc `0.8963`, macro `0.8867`
  epochs run before resample/stop: `2`
- loop `139` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9659`
  best loop real test: acc `0.8965`, macro `0.8869`
  epochs run before resample/stop: `1`
- loop `140` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8965`, macro `0.8869`
  epochs run before resample/stop: `1`
- loop `141` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8965`, macro `0.8869`
  epochs run before resample/stop: `1`
- loop `142` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9659`
  best loop real test: acc `0.8956`, macro `0.8859`
  epochs run before resample/stop: `2`
- loop `143` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9659`
  best loop real test: acc `0.8963`, macro `0.8867`
  epochs run before resample/stop: `1`
- loop `144` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8968`, macro `0.8873`
  epochs run before resample/stop: `2`
- loop `145` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8965`, macro `0.8869`
  epochs run before resample/stop: `2`
- loop `146` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8967`, macro `0.8871`
  epochs run before resample/stop: `1`
- loop `147` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8968`, macro `0.8873`
  epochs run before resample/stop: `1`
- loop `148` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9660`
  best loop real test: acc `0.8970`, macro `0.8876`
  epochs run before resample/stop: `2`
- loop `149` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8967`, macro `0.8871`
  epochs run before resample/stop: `1`
- loop `150` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9659`
  best loop real test: acc `0.8963`, macro `0.8867`
  epochs run before resample/stop: `1`
- loop `151` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8970`, macro `0.8874`
  epochs run before resample/stop: `2`
- loop `152` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `2`
- loop `153` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `1`
- loop `154` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `1`
- loop `155` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `1`
- loop `156` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `2`
- loop `157` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8965`, macro `0.8869`
  epochs run before resample/stop: `2`
- loop `158` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8968`, macro `0.8872`
  epochs run before resample/stop: `1`
- loop `159` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8965`, macro `0.8869`
  epochs run before resample/stop: `2`
- loop `160` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8967`, macro `0.8870`
  epochs run before resample/stop: `1`
- loop `161` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8968`, macro `0.8873`
  epochs run before resample/stop: `2`
- loop `162` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8968`, macro `0.8873`
  epochs run before resample/stop: `1`
- loop `163` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8965`, macro `0.8869`
  epochs run before resample/stop: `2`
- loop `164` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8963`, macro `0.8867`
  epochs run before resample/stop: `2`
- loop `165` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8967`, macro `0.8871`
  epochs run before resample/stop: `1`
- loop `166` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8967`, macro `0.8871`
  epochs run before resample/stop: `1`
- loop `167` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8963`, macro `0.8867`
  epochs run before resample/stop: `1`
- loop `168` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8968`, macro `0.8872`
  epochs run before resample/stop: `2`
- loop `169` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8965`, macro `0.8868`
  epochs run before resample/stop: `1`
- loop `170` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8974`, macro `0.8878`
  epochs run before resample/stop: `2`
- loop `171` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9659`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `2`
- loop `172` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9659`
  best loop real test: acc `0.8974`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `173` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9659`
  best loop real test: acc `0.8968`, macro `0.8872`
  epochs run before resample/stop: `1`
- loop `174` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8974`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `175` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `1`
- loop `176` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `2`
- loop `177` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9658`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `1`
- loop `178` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9658`
  best loop real test: acc `0.8967`, macro `0.8871`
  epochs run before resample/stop: `1`
- loop `179` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9658`
  best loop real test: acc `0.8967`, macro `0.8870`
  epochs run before resample/stop: `1`
- loop `180` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9658`
  best loop real test: acc `0.8967`, macro `0.8870`
  epochs run before resample/stop: `2`
- loop `181` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9658`
  best loop real test: acc `0.8970`, macro `0.8874`
  epochs run before resample/stop: `1`
- loop `182` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9658`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `1`
- loop `183` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `2`
- loop `184` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9658`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `1`
- loop `185` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `2`
- loop `186` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9658`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `1`
- loop `187` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9658`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `2`
- loop `188` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9658`
  best loop real test: acc `0.8977`, macro `0.8882`
  epochs run before resample/stop: `2`
- loop `189` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `2`
- loop `190` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `2`
- loop `191` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `1`
- loop `192` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9658`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `1`
- loop `193` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9658`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `194` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9658`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `2`
- loop `195` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8979`, macro `0.8885`
  epochs run before resample/stop: `2`
- loop `196` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8970`, macro `0.8874`
  epochs run before resample/stop: `1`
- loop `197` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8970`, macro `0.8874`
  epochs run before resample/stop: `2`
- loop `198` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9658`
  best loop real test: acc `0.8970`, macro `0.8874`
  epochs run before resample/stop: `1`
- loop `199` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9658`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `2`
- loop `200` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8967`, macro `0.8870`
  epochs run before resample/stop: `2`
- loop `201` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9658`
  best loop real test: acc `0.8968`, macro `0.8872`
  epochs run before resample/stop: `1`
- loop `202` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8974`, macro `0.8878`
  epochs run before resample/stop: `2`
- loop `203` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8977`, macro `0.8881`
  epochs run before resample/stop: `1`
- loop `204` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8977`, macro `0.8882`
  epochs run before resample/stop: `1`
- loop `205` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8977`, macro `0.8882`
  epochs run before resample/stop: `1`
- loop `206` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8974`, macro `0.8878`
  epochs run before resample/stop: `2`
- loop `207` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `2`
- loop `208` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8968`, macro `0.8872`
  epochs run before resample/stop: `2`
- loop `209` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8977`, macro `0.8882`
  epochs run before resample/stop: `2`
- loop `210` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `2`
- loop `211` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `2`
- loop `212` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `2`
- loop `213` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `1`
- loop `214` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `2`
- loop `215` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8974`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `216` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9661`
  best loop real test: acc `0.8979`, macro `0.8885`
  epochs run before resample/stop: `2`
- loop `217` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9661`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `1`
- loop `218` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9661`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `2`
- loop `219` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9661`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `1`
- loop `220` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `1`
- loop `221` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `2`
- loop `222` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8974`, macro `0.8880`
  epochs run before resample/stop: `2`
- loop `223` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `1`
- loop `224` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `1`
- loop `225` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8967`, macro `0.8871`
  epochs run before resample/stop: `2`
- loop `226` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `1`
- loop `227` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8967`, macro `0.8871`
  epochs run before resample/stop: `1`
- loop `228` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `1`
- loop `229` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `1`
- loop `230` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8967`, macro `0.8870`
  epochs run before resample/stop: `2`
- loop `231` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8968`, macro `0.8872`
  epochs run before resample/stop: `2`
- loop `232` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8968`, macro `0.8873`
  epochs run before resample/stop: `2`
- loop `233` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `2`
- loop `234` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `1`
- loop `235` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8981`, macro `0.8887`
  epochs run before resample/stop: `1`
- loop `236` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `1`
- loop `237` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `1`
- loop `238` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8975`, macro `0.8882`
  epochs run before resample/stop: `1`
- loop `239` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9661`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `2`
- loop `240` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `1`
- loop `241` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `1`
- loop `242` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8979`, macro `0.8885`
  epochs run before resample/stop: `1`
- loop `243` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `2`
- loop `244` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `1`
- loop `245` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8979`, macro `0.8885`
  epochs run before resample/stop: `1`
- loop `246` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `1`
- loop `247` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `2`
- loop `248` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8970`, macro `0.8874`
  epochs run before resample/stop: `2`
- loop `249` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8975`, macro `0.8880`
  epochs run before resample/stop: `1`
- loop `250` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8977`, macro `0.8882`
  epochs run before resample/stop: `1`
- loop `251` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8974`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `252` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8974`, macro `0.8878`
  epochs run before resample/stop: `2`
- loop `253` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8981`, macro `0.8887`
  epochs run before resample/stop: `2`
- loop `254` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `1`
- loop `255` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8970`, macro `0.8874`
  epochs run before resample/stop: `1`
- loop `256` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8975`, macro `0.8880`
  epochs run before resample/stop: `1`
- loop `257` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `1`
- loop `258` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `1`
- loop `259` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8981`, macro `0.8887`
  epochs run before resample/stop: `2`
- loop `260` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8974`, macro `0.8878`
  epochs run before resample/stop: `2`
- loop `261` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `1`
- loop `262` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `2`
- loop `263` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9659`
  best loop real test: acc `0.8981`, macro `0.8887`
  epochs run before resample/stop: `2`
- loop `264` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9659`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `1`
- loop `265` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `2`
- loop `266` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `1`
- loop `267` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8975`, macro `0.8880`
  epochs run before resample/stop: `1`
- loop `268` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9658`
  best loop real test: acc `0.8968`, macro `0.8873`
  epochs run before resample/stop: `1`
- loop `269` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8967`, macro `0.8871`
  epochs run before resample/stop: `2`
- loop `270` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `2`
- loop `271` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `2`
- loop `272` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `1`
- loop `273` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8979`, macro `0.8884`
  epochs run before resample/stop: `2`
- loop `274` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8975`, macro `0.8880`
  epochs run before resample/stop: `1`
- loop `275` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8977`, macro `0.8882`
  epochs run before resample/stop: `2`
- loop `276` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8981`, macro `0.8887`
  epochs run before resample/stop: `2`
- loop `277` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8981`, macro `0.8887`
  epochs run before resample/stop: `1`
- loop `278` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9661`
  best loop real test: acc `0.8983`, macro `0.8889`
  epochs run before resample/stop: `2`
- loop `279` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9660`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `1`
- loop `280` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9660`
  best loop real test: acc `0.8977`, macro `0.8884`
  epochs run before resample/stop: `1`
- loop `281` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `1`
- loop `282` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `1`
- loop `283` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8967`, macro `0.8871`
  epochs run before resample/stop: `1`
- loop `284` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `2`
- loop `285` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `1`
- loop `286` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `1`
- loop `287` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8979`, macro `0.8886`
  epochs run before resample/stop: `2`
- loop `288` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `1`
- loop `289` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `2`
- loop `290` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `1`
- loop `291` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `2`
- loop `292` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `1`
- loop `293` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `1`
- loop `294` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8981`, macro `0.8887`
  epochs run before resample/stop: `2`
- loop `295` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `1`
- loop `296` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `1`
- loop `297` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `2`
- loop `298` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `2`
- loop `299` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `1`
- loop `300` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8981`, macro `0.8887`
  epochs run before resample/stop: `1`
- loop `301` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8981`, macro `0.8887`
  epochs run before resample/stop: `2`
- loop `302` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8984`, macro `0.8891`
  epochs run before resample/stop: `1`
- loop `303` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8986`, macro `0.8893`
  epochs run before resample/stop: `1`
- loop `304` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8979`, macro `0.8885`
  epochs run before resample/stop: `1`
- loop `305` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `2`
- loop `306` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8983`, macro `0.8889`
  epochs run before resample/stop: `2`
- loop `307` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8979`, macro `0.8885`
  epochs run before resample/stop: `1`
- loop `308` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `1`
- loop `309` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9658`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `1`
- loop `310` accepted: `True`
  best synthetic val score: macro `0.9648`, acc `0.9658`
  best loop real test: acc `0.8979`, macro `0.8885`
  epochs run before resample/stop: `1`
- loop `311` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9658`
  best loop real test: acc `0.8983`, macro `0.8889`
  epochs run before resample/stop: `2`
- loop `312` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9658`
  best loop real test: acc `0.8979`, macro `0.8885`
  epochs run before resample/stop: `2`
- loop `313` accepted: `True`
  best synthetic val score: macro `0.9648`, acc `0.9658`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `1`
- loop `314` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `2`
- loop `315` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `1`
- loop `316` accepted: `True`
  best synthetic val score: macro `0.9648`, acc `0.9658`
  best loop real test: acc `0.8975`, macro `0.8882`
  epochs run before resample/stop: `1`
- loop `317` accepted: `True`
  best synthetic val score: macro `0.9648`, acc `0.9658`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `1`
- loop `318` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9658`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `2`
- loop `319` accepted: `True`
  best synthetic val score: macro `0.9648`, acc `0.9658`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `1`
- loop `320` accepted: `True`
  best synthetic val score: macro `0.9648`, acc `0.9658`
  best loop real test: acc `0.8974`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `321` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9658`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `2`
- loop `322` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9658`
  best loop real test: acc `0.8970`, macro `0.8874`
  epochs run before resample/stop: `2`
- loop `323` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9658`
  best loop real test: acc `0.8974`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `324` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9658`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `1`
- loop `325` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `2`
- loop `326` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `1`
- loop `327` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `2`
- loop `328` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `1`
- loop `329` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8974`, macro `0.8878`
  epochs run before resample/stop: `2`
- loop `330` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `1`
- loop `331` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `1`
- loop `332` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `1`
- loop `333` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `2`
- loop `334` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `1`
- loop `335` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `336` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `1`
- loop `337` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `2`
- loop `338` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `1`
- loop `339` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `1`
- loop `340` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `1`
- loop `341` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `1`
- loop `342` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `2`
- loop `343` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `2`
- loop `344` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8979`, macro `0.8885`
  epochs run before resample/stop: `2`
- loop `345` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8979`, macro `0.8885`
  epochs run before resample/stop: `1`
- loop `346` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `2`
- loop `347` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `1`
- loop `348` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `2`
- loop `349` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `1`
- loop `350` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `1`
- loop `351` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `1`
- loop `352` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8975`, macro `0.8882`
  epochs run before resample/stop: `1`
- loop `353` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `2`
- loop `354` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `2`
- loop `355` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `1`
- loop `356` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `2`
- loop `357` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `2`
- loop `358` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `1`
- loop `359` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8974`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `360` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `2`
- loop `361` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `1`
- loop `362` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8970`, macro `0.8874`
  epochs run before resample/stop: `1`
- loop `363` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8974`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `364` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `365` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8968`, macro `0.8873`
  epochs run before resample/stop: `2`
- loop `366` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `1`
- loop `367` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `2`
- loop `368` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `2`
- loop `369` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `1`
- loop `370` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `2`
- loop `371` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `1`
- loop `372` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `1`
- loop `373` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `1`
- loop `374` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `1`
- loop `375` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8967`, macro `0.8871`
  epochs run before resample/stop: `1`
- loop `376` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8968`, macro `0.8872`
  epochs run before resample/stop: `2`
- loop `377` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `2`
- loop `378` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `2`
- loop `379` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `2`
- loop `380` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `1`
- loop `381` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8970`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `382` accepted: `True`
  best synthetic val score: macro `0.9648`, acc `0.9659`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `1`
- loop `383` accepted: `True`
  best synthetic val score: macro `0.9648`, acc `0.9658`
  best loop real test: acc `0.8968`, macro `0.8873`
  epochs run before resample/stop: `1`
- loop `384` accepted: `True`
  best synthetic val score: macro `0.9648`, acc `0.9658`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `2`
- loop `385` accepted: `True`
  best synthetic val score: macro `0.9648`, acc `0.9658`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `1`
- loop `386` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `2`
- loop `387` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9658`
  best loop real test: acc `0.8968`, macro `0.8873`
  epochs run before resample/stop: `1`
- loop `388` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9658`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `1`
- loop `389` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `2`
- loop `390` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `2`
- loop `391` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `1`
- loop `392` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8981`, macro `0.8888`
  epochs run before resample/stop: `2`
- loop `393` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8981`, macro `0.8887`
  epochs run before resample/stop: `2`
- loop `394` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8986`, macro `0.8893`
  epochs run before resample/stop: `1`
- loop `395` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8983`, macro `0.8889`
  epochs run before resample/stop: `2`
- loop `396` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `2`
- loop `397` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8975`, macro `0.8882`
  epochs run before resample/stop: `1`
- loop `398` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `2`
- loop `399` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8979`, macro `0.8885`
  epochs run before resample/stop: `1`
- loop `400` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8974`, macro `0.8880`
  epochs run before resample/stop: `2`
- loop `401` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8981`, macro `0.8888`
  epochs run before resample/stop: `1`
- loop `402` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8981`, macro `0.8888`
  epochs run before resample/stop: `2`
- loop `403` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8977`, macro `0.8884`
  epochs run before resample/stop: `2`
- loop `404` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `1`
- loop `405` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `2`
- loop `406` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `2`
- loop `407` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `1`
- loop `408` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8977`, macro `0.8884`
  epochs run before resample/stop: `1`
- loop `409` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `1`
- loop `410` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8983`, macro `0.8890`
  epochs run before resample/stop: `2`
- loop `411` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `1`
- loop `412` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `1`
- loop `413` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `2`
- loop `414` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8975`, macro `0.8882`
  epochs run before resample/stop: `1`
- loop `415` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `2`
- loop `416` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8975`, macro `0.8882`
  epochs run before resample/stop: `1`
- loop `417` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8970`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `418` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8979`, macro `0.8885`
  epochs run before resample/stop: `2`
- loop `419` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8979`, macro `0.8885`
  epochs run before resample/stop: `1`
- loop `420` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `1`
- loop `421` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `1`
- loop `422` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `1`
- loop `423` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `1`
- loop `424` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `2`
- loop `425` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `2`
- loop `426` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `2`
- loop `427` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9662`
  best loop real test: acc `0.8981`, macro `0.8888`
  epochs run before resample/stop: `2`
- loop `428` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8984`, macro `0.8892`
  epochs run before resample/stop: `1`
- loop `429` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `2`
- loop `430` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8977`, macro `0.8882`
  epochs run before resample/stop: `1`
- loop `431` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `2`
- loop `432` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9661`
  best loop real test: acc `0.8979`, macro `0.8885`
  epochs run before resample/stop: `2`
- loop `433` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9661`
  best loop real test: acc `0.8981`, macro `0.8887`
  epochs run before resample/stop: `1`
- loop `434` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9662`
  best loop real test: acc `0.8981`, macro `0.8887`
  epochs run before resample/stop: `2`
- loop `435` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8984`, macro `0.8891`
  epochs run before resample/stop: `1`
- loop `436` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `1`
- loop `437` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9662`
  best loop real test: acc `0.8975`, macro `0.8880`
  epochs run before resample/stop: `2`
- loop `438` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9662`
  best loop real test: acc `0.8979`, macro `0.8885`
  epochs run before resample/stop: `1`
- loop `439` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9662`
  best loop real test: acc `0.8979`, macro `0.8884`
  epochs run before resample/stop: `2`
- loop `440` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8977`, macro `0.8882`
  epochs run before resample/stop: `1`
- loop `441` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8974`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `442` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8974`, macro `0.8877`
  epochs run before resample/stop: `1`
- loop `443` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9661`
  best loop real test: acc `0.8975`, macro `0.8880`
  epochs run before resample/stop: `2`
- loop `444` accepted: `True`
  best synthetic val score: macro `0.9653`, acc `0.9662`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `2`
- loop `445` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9662`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `1`
- loop `446` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9662`
  best loop real test: acc `0.8979`, macro `0.8885`
  epochs run before resample/stop: `2`
- loop `447` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9662`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `1`
- loop `448` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9662`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `1`
- loop `449` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9661`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `1`
- loop `450` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `1`
- loop `451` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9662`
  best loop real test: acc `0.8979`, macro `0.8885`
  epochs run before resample/stop: `2`
- loop `452` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9662`
  best loop real test: acc `0.8977`, macro `0.8882`
  epochs run before resample/stop: `1`
- loop `453` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9662`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `1`
- loop `454` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `1`
- loop `455` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8979`, macro `0.8884`
  epochs run before resample/stop: `1`
- loop `456` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `2`
- loop `457` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8975`, macro `0.8880`
  epochs run before resample/stop: `2`
- loop `458` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9661`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `2`
- loop `459` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `1`
- loop `460` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9661`
  best loop real test: acc `0.8977`, macro `0.8884`
  epochs run before resample/stop: `2`
- loop `461` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9662`
  best loop real test: acc `0.8977`, macro `0.8884`
  epochs run before resample/stop: `2`
- loop `462` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9661`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `1`
- loop `463` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9661`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `2`
- loop `464` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `1`
- loop `465` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `1`
- loop `466` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `1`
- loop `467` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8974`, macro `0.8880`
  epochs run before resample/stop: `2`
- loop `468` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `2`
- loop `469` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `2`
- loop `470` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8975`, macro `0.8882`
  epochs run before resample/stop: `1`
- loop `471` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8975`, macro `0.8882`
  epochs run before resample/stop: `1`
- loop `472` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8974`, macro `0.8880`
  epochs run before resample/stop: `1`
- loop `473` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8975`, macro `0.8882`
  epochs run before resample/stop: `2`
- loop `474` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8977`, macro `0.8884`
  epochs run before resample/stop: `1`
- loop `475` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8977`, macro `0.8882`
  epochs run before resample/stop: `2`
- loop `476` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `1`
- loop `477` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `1`
- loop `478` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `1`
- loop `479` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `1`
- loop `480` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8974`, macro `0.8880`
  epochs run before resample/stop: `1`
- loop `481` accepted: `True`
  best synthetic val score: macro `0.9648`, acc `0.9659`
  best loop real test: acc `0.8967`, macro `0.8871`
  epochs run before resample/stop: `1`
- loop `482` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `2`
- loop `483` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8979`, macro `0.8885`
  epochs run before resample/stop: `1`
- loop `484` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `1`
- loop `485` accepted: `True`
  best synthetic val score: macro `0.9648`, acc `0.9660`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `2`
- loop `486` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `2`
- loop `487` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `1`
- loop `488` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `1`
- loop `489` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `2`
- loop `490` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `2`
- loop `491` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8970`, macro `0.8876`
  epochs run before resample/stop: `2`
- loop `492` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `1`
- loop `493` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8975`, macro `0.8882`
  epochs run before resample/stop: `2`
- loop `494` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8983`, macro `0.8889`
  epochs run before resample/stop: `2`
- loop `495` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `1`
- loop `496` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `1`
- loop `497` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8979`, macro `0.8885`
  epochs run before resample/stop: `2`
- loop `498` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8977`, macro `0.8884`
  epochs run before resample/stop: `1`
- loop `499` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8979`, macro `0.8885`
  epochs run before resample/stop: `1`
- loop `500` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `1`
- loop `501` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `1`
- loop `502` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8981`, macro `0.8887`
  epochs run before resample/stop: `2`
- loop `503` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `1`
- loop `504` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `1`
- loop `505` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `2`
- loop `506` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `2`
- loop `507` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8981`, macro `0.8886`
  epochs run before resample/stop: `2`
- loop `508` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8981`, macro `0.8887`
  epochs run before resample/stop: `2`
- loop `509` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8983`, macro `0.8889`
  epochs run before resample/stop: `1`
- loop `510` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8979`, macro `0.8885`
  epochs run before resample/stop: `2`
- loop `511` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8983`, macro `0.8889`
  epochs run before resample/stop: `1`
- loop `512` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8979`, macro `0.8885`
  epochs run before resample/stop: `2`
- loop `513` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8977`, macro `0.8884`
  epochs run before resample/stop: `1`
- loop `514` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8974`, macro `0.8880`
  epochs run before resample/stop: `2`
- loop `515` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `2`
- loop `516` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8974`, macro `0.8880`
  epochs run before resample/stop: `2`
- loop `517` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8970`, macro `0.8876`
  epochs run before resample/stop: `2`
- loop `518` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `2`
- loop `519` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `1`
- loop `520` accepted: `True`
  best synthetic val score: macro `0.9648`, acc `0.9659`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `1`
- loop `521` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8979`, macro `0.8886`
  epochs run before resample/stop: `2`
- loop `522` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `2`
- loop `523` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `2`
- loop `524` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8967`, macro `0.8871`
  epochs run before resample/stop: `2`
- loop `525` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8968`, macro `0.8873`
  epochs run before resample/stop: `1`
- loop `526` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8968`, macro `0.8873`
  epochs run before resample/stop: `2`
- loop `527` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `2`
- loop `528` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8968`, macro `0.8873`
  epochs run before resample/stop: `1`
- loop `529` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `1`
- loop `530` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8979`, macro `0.8885`
  epochs run before resample/stop: `2`
- loop `531` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `1`
- loop `532` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8979`, macro `0.8885`
  epochs run before resample/stop: `2`
- loop `533` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `2`
- loop `534` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `2`
- loop `535` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8975`, macro `0.8880`
  epochs run before resample/stop: `1`
- loop `536` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `2`
- loop `537` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8974`, macro `0.8880`
  epochs run before resample/stop: `1`
- loop `538` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `1`
- loop `539` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `1`
- loop `540` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `2`
- loop `541` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8968`, macro `0.8873`
  epochs run before resample/stop: `2`
- loop `542` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `1`
- loop `543` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9661`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `2`
- loop `544` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9661`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `2`
- loop `545` accepted: `True`
  best synthetic val score: macro `0.9648`, acc `0.9660`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `1`
- loop `546` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `2`
- loop `547` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `2`
- loop `548` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9662`
  best loop real test: acc `0.8974`, macro `0.8880`
  epochs run before resample/stop: `2`
- loop `549` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8979`, macro `0.8886`
  epochs run before resample/stop: `1`
- loop `550` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `2`
- loop `551` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `2`
- loop `552` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9662`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `2`
- loop `553` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9662`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `1`
- loop `554` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `2`
- loop `555` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `1`
- loop `556` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `1`
- loop `557` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9662`
  best loop real test: acc `0.8979`, macro `0.8885`
  epochs run before resample/stop: `2`
- loop `558` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `1`
- loop `559` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8977`, macro `0.8884`
  epochs run before resample/stop: `1`
- loop `560` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9662`
  best loop real test: acc `0.8979`, macro `0.8885`
  epochs run before resample/stop: `2`
- loop `561` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8979`, macro `0.8885`
  epochs run before resample/stop: `1`
- loop `562` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `1`
- loop `563` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `2`
- loop `564` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `1`
- loop `565` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `1`
- loop `566` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `2`
- loop `567` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8979`, macro `0.8885`
  epochs run before resample/stop: `1`
- loop `568` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8979`, macro `0.8885`
  epochs run before resample/stop: `2`
- loop `569` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8979`, macro `0.8886`
  epochs run before resample/stop: `2`
- loop `570` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8974`, macro `0.8880`
  epochs run before resample/stop: `2`
- loop `571` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `2`
- loop `572` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `2`
- loop `573` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `2`
- loop `574` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `2`
- loop `575` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9662`
  best loop real test: acc `0.8977`, macro `0.8884`
  epochs run before resample/stop: `2`
- loop `576` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `1`
- loop `577` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `1`
- loop `578` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `1`
- loop `579` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8979`, macro `0.8884`
  epochs run before resample/stop: `1`
- loop `580` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8977`, macro `0.8882`
  epochs run before resample/stop: `2`
- loop `581` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8977`, macro `0.8882`
  epochs run before resample/stop: `2`
- loop `582` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8979`, macro `0.8884`
  epochs run before resample/stop: `2`
- loop `583` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8979`, macro `0.8884`
  epochs run before resample/stop: `1`
- loop `584` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9662`
  best loop real test: acc `0.8981`, macro `0.8887`
  epochs run before resample/stop: `2`
- loop `585` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8979`, macro `0.8884`
  epochs run before resample/stop: `2`
- loop `586` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8981`, macro `0.8886`
  epochs run before resample/stop: `1`
- loop `587` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8974`, macro `0.8878`
  epochs run before resample/stop: `2`
- loop `588` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `2`
- loop `589` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `1`
- loop `590` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `2`
- loop `591` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `2`
- loop `592` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8967`, macro `0.8871`
  epochs run before resample/stop: `2`
- loop `593` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8968`, macro `0.8873`
  epochs run before resample/stop: `1`
- loop `594` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `1`
- loop `595` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8968`, macro `0.8873`
  epochs run before resample/stop: `2`
- loop `596` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8967`, macro `0.8871`
  epochs run before resample/stop: `1`
- loop `597` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `2`
- loop `598` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8968`, macro `0.8872`
  epochs run before resample/stop: `1`
- loop `599` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8968`, macro `0.8872`
  epochs run before resample/stop: `1`
- loop `600` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8975`, macro `0.8880`
  epochs run before resample/stop: `2`
- loop `601` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `2`
- loop `602` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `2`
- loop `603` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `1`
- loop `604` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `1`
- loop `605` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `2`
- loop `606` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `1`
- loop `607` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `2`
- loop `608` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `1`
- loop `609` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `2`
- loop `610` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8979`, macro `0.8885`
  epochs run before resample/stop: `2`
- loop `611` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8977`, macro `0.8882`
  epochs run before resample/stop: `1`
- loop `612` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `2`
- loop `613` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8974`, macro `0.8878`
  epochs run before resample/stop: `2`
- loop `614` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8974`, macro `0.8878`
  epochs run before resample/stop: `2`
- loop `615` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8975`, macro `0.8880`
  epochs run before resample/stop: `1`
- loop `616` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8968`, macro `0.8873`
  epochs run before resample/stop: `2`
- loop `617` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8968`, macro `0.8873`
  epochs run before resample/stop: `1`
- loop `618` accepted: `True`
  best synthetic val score: macro `0.9648`, acc `0.9659`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `1`
- loop `619` accepted: `True`
  best synthetic val score: macro `0.9648`, acc `0.9659`
  best loop real test: acc `0.8970`, macro `0.8874`
  epochs run before resample/stop: `1`
- loop `620` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `2`
- loop `621` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `2`
- loop `622` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8974`, macro `0.8880`
  epochs run before resample/stop: `2`
- loop `623` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `2`
- loop `624` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `2`
- loop `625` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `1`
- loop `626` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `1`
- loop `627` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `2`
- loop `628` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8974`, macro `0.8880`
  epochs run before resample/stop: `2`
- loop `629` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `1`
- loop `630` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `2`
- loop `631` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8970`, macro `0.8876`
  epochs run before resample/stop: `2`
- loop `632` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `2`
- loop `633` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `1`
- loop `634` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `2`
- loop `635` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8974`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `636` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `2`
- loop `637` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `1`
- loop `638` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8983`, macro `0.8890`
  epochs run before resample/stop: `2`
- loop `639` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8977`, macro `0.8884`
  epochs run before resample/stop: `2`
- loop `640` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `1`
- loop `641` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `1`
- loop `642` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `2`
- loop `643` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `2`
- loop `644` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `1`
- loop `645` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8968`, macro `0.8873`
  epochs run before resample/stop: `2`
- loop `646` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8965`, macro `0.8869`
  epochs run before resample/stop: `1`
- loop `647` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8967`, macro `0.8871`
  epochs run before resample/stop: `2`
- loop `648` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8968`, macro `0.8873`
  epochs run before resample/stop: `1`
- loop `649` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8961`, macro `0.8866`
  epochs run before resample/stop: `2`
- loop `650` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `2`
- loop `651` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8968`, macro `0.8873`
  epochs run before resample/stop: `2`
- loop `652` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8974`, macro `0.8880`
  epochs run before resample/stop: `2`
- loop `653` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8970`, macro `0.8876`
  epochs run before resample/stop: `2`
- loop `654` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8967`, macro `0.8871`
  epochs run before resample/stop: `1`
- loop `655` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8968`, macro `0.8873`
  epochs run before resample/stop: `2`
- loop `656` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8970`, macro `0.8874`
  epochs run before resample/stop: `2`
- loop `657` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `2`
- loop `658` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `2`
- loop `659` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9662`
  best loop real test: acc `0.8983`, macro `0.8889`
  epochs run before resample/stop: `2`
- loop `660` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9662`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `2`
- loop `661` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9662`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `1`
- loop `662` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9662`
  best loop real test: acc `0.8979`, macro `0.8885`
  epochs run before resample/stop: `1`
- loop `663` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9662`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `2`
- loop `664` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9662`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `1`
- loop `665` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9662`
  best loop real test: acc `0.8983`, macro `0.8889`
  epochs run before resample/stop: `2`
- loop `666` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9662`
  best loop real test: acc `0.8981`, macro `0.8887`
  epochs run before resample/stop: `1`
- loop `667` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9662`
  best loop real test: acc `0.8981`, macro `0.8887`
  epochs run before resample/stop: `1`
- loop `668` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9662`
  best loop real test: acc `0.8981`, macro `0.8887`
  epochs run before resample/stop: `1`
- loop `669` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `1`
- loop `670` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `1`
- loop `671` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `2`
- loop `672` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `2`
- loop `673` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `1`
- loop `674` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9661`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `2`
- loop `675` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8979`, macro `0.8885`
  epochs run before resample/stop: `1`
- loop `676` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9661`
  best loop real test: acc `0.8981`, macro `0.8887`
  epochs run before resample/stop: `2`
- loop `677` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9661`
  best loop real test: acc `0.8984`, macro `0.8891`
  epochs run before resample/stop: `1`
- loop `678` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8983`, macro `0.8888`
  epochs run before resample/stop: `1`
- loop `679` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `1`
- loop `680` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8979`, macro `0.8885`
  epochs run before resample/stop: `2`
- loop `681` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8981`, macro `0.8887`
  epochs run before resample/stop: `1`
- loop `682` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `2`
- loop `683` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8970`, macro `0.8874`
  epochs run before resample/stop: `1`
- loop `684` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `2`
- loop `685` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8981`, macro `0.8887`
  epochs run before resample/stop: `2`
- loop `686` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `2`
- loop `687` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `1`
- loop `688` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `2`
- loop `689` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `1`
- loop `690` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `2`
- loop `691` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8974`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `692` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `693` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8975`, macro `0.8880`
  epochs run before resample/stop: `2`
- loop `694` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `2`
- loop `695` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8981`, macro `0.8887`
  epochs run before resample/stop: `2`
- loop `696` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8974`, macro `0.8880`
  epochs run before resample/stop: `2`
- loop `697` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `2`
- loop `698` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `2`
- loop `699` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `1`
- loop `700` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8975`, macro `0.8882`
  epochs run before resample/stop: `1`
- loop `701` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8981`, macro `0.8887`
  epochs run before resample/stop: `2`
- loop `702` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8977`, macro `0.8884`
  epochs run before resample/stop: `1`
- loop `703` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8977`, macro `0.8884`
  epochs run before resample/stop: `1`
- loop `704` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8975`, macro `0.8882`
  epochs run before resample/stop: `2`
- loop `705` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `1`
- loop `706` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8975`, macro `0.8882`
  epochs run before resample/stop: `1`
- loop `707` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8974`, macro `0.8880`
  epochs run before resample/stop: `1`
- loop `708` accepted: `True`
  best synthetic val score: macro `0.9648`, acc `0.9659`
  best loop real test: acc `0.8977`, macro `0.8884`
  epochs run before resample/stop: `2`
- loop `709` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9658`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `2`
- loop `710` accepted: `True`
  best synthetic val score: macro `0.9648`, acc `0.9659`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `2`
- loop `711` accepted: `True`
  best synthetic val score: macro `0.9648`, acc `0.9659`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `1`
- loop `712` accepted: `True`
  best synthetic val score: macro `0.9648`, acc `0.9658`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `2`
- loop `713` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `2`
- loop `714` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `1`
- loop `715` accepted: `True`
  best synthetic val score: macro `0.9648`, acc `0.9658`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `1`
- loop `716` accepted: `True`
  best synthetic val score: macro `0.9648`, acc `0.9658`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `1`
- loop `717` accepted: `True`
  best synthetic val score: macro `0.9648`, acc `0.9658`
  best loop real test: acc `0.8983`, macro `0.8888`
  epochs run before resample/stop: `2`
- loop `718` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8981`, macro `0.8886`
  epochs run before resample/stop: `2`
- loop `719` accepted: `True`
  best synthetic val score: macro `0.9648`, acc `0.9659`
  best loop real test: acc `0.8983`, macro `0.8889`
  epochs run before resample/stop: `1`
- loop `720` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8983`, macro `0.8889`
  epochs run before resample/stop: `2`
- loop `721` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8981`, macro `0.8887`
  epochs run before resample/stop: `2`
- loop `722` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `2`
- loop `723` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8981`, macro `0.8888`
  epochs run before resample/stop: `1`
- loop `724` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8977`, macro `0.8884`
  epochs run before resample/stop: `1`
- loop `725` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8977`, macro `0.8884`
  epochs run before resample/stop: `2`
- loop `726` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `2`
- loop `727` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `1`
- loop `728` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8974`, macro `0.8880`
  epochs run before resample/stop: `1`
- loop `729` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `2`
- loop `730` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `2`
- loop `731` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `1`
- loop `732` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `1`
- loop `733` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8979`, macro `0.8885`
  epochs run before resample/stop: `2`
- loop `734` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8979`, macro `0.8886`
  epochs run before resample/stop: `2`
- loop `735` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `1`
- loop `736` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8977`, macro `0.8884`
  epochs run before resample/stop: `2`
- loop `737` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8981`, macro `0.8887`
  epochs run before resample/stop: `2`
- loop `738` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8977`, macro `0.8884`
  epochs run before resample/stop: `1`
- loop `739` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `2`
- loop `740` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9658`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `2`
- loop `741` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9658`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `1`
- loop `742` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8975`, macro `0.8882`
  epochs run before resample/stop: `2`
- loop `743` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `1`
- loop `744` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8974`, macro `0.8880`
  epochs run before resample/stop: `2`
- loop `745` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8977`, macro `0.8884`
  epochs run before resample/stop: `2`
- loop `746` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `1`
- loop `747` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `1`
- loop `748` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `2`
- loop `749` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `2`
- loop `750` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `1`
- loop `751` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `2`
- loop `752` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `1`
- loop `753` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `2`
- loop `754` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `1`
- loop `755` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9661`
  best loop real test: acc `0.8967`, macro `0.8871`
  epochs run before resample/stop: `2`
- loop `756` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9661`
  best loop real test: acc `0.8965`, macro `0.8870`
  epochs run before resample/stop: `1`
- loop `757` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9661`
  best loop real test: acc `0.8967`, macro `0.8872`
  epochs run before resample/stop: `2`
- loop `758` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9661`
  best loop real test: acc `0.8968`, macro `0.8874`
  epochs run before resample/stop: `2`
- loop `759` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9661`
  best loop real test: acc `0.8968`, macro `0.8874`
  epochs run before resample/stop: `2`
- loop `760` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9661`
  best loop real test: acc `0.8967`, macro `0.8872`
  epochs run before resample/stop: `1`
- loop `761` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8965`, macro `0.8870`
  epochs run before resample/stop: `1`
- loop `762` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8967`, macro `0.8872`
  epochs run before resample/stop: `1`
- loop `763` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9662`
  best loop real test: acc `0.8967`, macro `0.8872`
  epochs run before resample/stop: `2`
- loop `764` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9662`
  best loop real test: acc `0.8970`, macro `0.8876`
  epochs run before resample/stop: `2`
- loop `765` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8968`, macro `0.8874`
  epochs run before resample/stop: `1`
- loop `766` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9661`
  best loop real test: acc `0.8970`, macro `0.8876`
  epochs run before resample/stop: `2`
- loop `767` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9661`
  best loop real test: acc `0.8967`, macro `0.8872`
  epochs run before resample/stop: `1`
- loop `768` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8965`, macro `0.8870`
  epochs run before resample/stop: `1`
- loop `769` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8965`, macro `0.8869`
  epochs run before resample/stop: `1`
- loop `770` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `2`
- loop `771` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8967`, macro `0.8872`
  epochs run before resample/stop: `1`
- loop `772` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8968`, macro `0.8874`
  epochs run before resample/stop: `2`
- loop `773` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8968`, macro `0.8874`
  epochs run before resample/stop: `2`
- loop `774` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8963`, macro `0.8867`
  epochs run before resample/stop: `1`
- loop `775` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8967`, macro `0.8870`
  epochs run before resample/stop: `2`
- loop `776` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8961`, macro `0.8864`
  epochs run before resample/stop: `1`
- loop `777` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8963`, macro `0.8867`
  epochs run before resample/stop: `2`
- loop `778` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8967`, macro `0.8871`
  epochs run before resample/stop: `2`
- loop `779` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8967`, macro `0.8872`
  epochs run before resample/stop: `2`
- loop `780` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8967`, macro `0.8872`
  epochs run before resample/stop: `2`
- loop `781` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8968`, macro `0.8874`
  epochs run before resample/stop: `2`
- loop `782` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8968`, macro `0.8874`
  epochs run before resample/stop: `1`
- loop `783` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8965`, macro `0.8870`
  epochs run before resample/stop: `2`
- loop `784` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9662`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `2`
- loop `785` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `786` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8970`, macro `0.8876`
  epochs run before resample/stop: `2`
- loop `787` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9662`
  best loop real test: acc `0.8968`, macro `0.8874`
  epochs run before resample/stop: `2`
- loop `788` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8967`, macro `0.8871`
  epochs run before resample/stop: `1`
- loop `789` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8967`, macro `0.8871`
  epochs run before resample/stop: `2`
- loop `790` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `2`
- loop `791` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `1`
- loop `792` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8984`, macro `0.8891`
  epochs run before resample/stop: `2`
- loop `793` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `1`
- loop `794` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `2`
- loop `795` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8984`, macro `0.8891`
  epochs run before resample/stop: `2`
- loop `796` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8977`, macro `0.8884`
  epochs run before resample/stop: `2`
- loop `797` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8983`, macro `0.8890`
  epochs run before resample/stop: `1`
- loop `798` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8984`, macro `0.8892`
  epochs run before resample/stop: `2`
- loop `799` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `1`
- loop `800` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8975`, macro `0.8880`
  epochs run before resample/stop: `1`
- loop `801` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8979`, macro `0.8885`
  epochs run before resample/stop: `1`
- loop `802` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8977`, macro `0.8882`
  epochs run before resample/stop: `1`
- loop `803` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `1`
- loop `804` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8876`
  epochs run before resample/stop: `2`
- loop `805` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8983`, macro `0.8889`
  epochs run before resample/stop: `2`
- loop `806` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8981`, macro `0.8887`
  epochs run before resample/stop: `2`
- loop `807` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `1`
- loop `808` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8979`, macro `0.8885`
  epochs run before resample/stop: `1`
- loop `809` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8977`, macro `0.8884`
  epochs run before resample/stop: `1`
- loop `810` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8979`, macro `0.8885`
  epochs run before resample/stop: `1`
- loop `811` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8977`, macro `0.8884`
  epochs run before resample/stop: `2`
- loop `812` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `1`
- loop `813` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8981`, macro `0.8887`
  epochs run before resample/stop: `2`
- loop `814` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9662`
  best loop real test: acc `0.8977`, macro `0.8884`
  epochs run before resample/stop: `2`
- loop `815` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9662`
  best loop real test: acc `0.8981`, macro `0.8887`
  epochs run before resample/stop: `1`
- loop `816` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9662`
  best loop real test: acc `0.8981`, macro `0.8888`
  epochs run before resample/stop: `2`
- loop `817` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9662`
  best loop real test: acc `0.8983`, macro `0.8890`
  epochs run before resample/stop: `1`
- loop `818` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8979`, macro `0.8886`
  epochs run before resample/stop: `1`
- loop `819` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8975`, macro `0.8882`
  epochs run before resample/stop: `1`
- loop `820` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9662`
  best loop real test: acc `0.8974`, macro `0.8880`
  epochs run before resample/stop: `1`
- loop `821` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8977`, macro `0.8884`
  epochs run before resample/stop: `1`
- loop `822` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8974`, macro `0.8880`
  epochs run before resample/stop: `1`
- loop `823` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8977`, macro `0.8884`
  epochs run before resample/stop: `2`
- loop `824` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9662`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `2`
- loop `825` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9662`
  best loop real test: acc `0.8977`, macro `0.8884`
  epochs run before resample/stop: `2`
- loop `826` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9662`
  best loop real test: acc `0.8981`, macro `0.8888`
  epochs run before resample/stop: `2`
- loop `827` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9662`
  best loop real test: acc `0.8981`, macro `0.8888`
  epochs run before resample/stop: `1`
- loop `828` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8981`, macro `0.8888`
  epochs run before resample/stop: `1`
- loop `829` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8979`, macro `0.8885`
  epochs run before resample/stop: `2`
- loop `830` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8974`, macro `0.8880`
  epochs run before resample/stop: `2`
- loop `831` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8979`, macro `0.8885`
  epochs run before resample/stop: `1`
- loop `832` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9662`
  best loop real test: acc `0.8983`, macro `0.8888`
  epochs run before resample/stop: `2`
- loop `833` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8984`, macro `0.8891`
  epochs run before resample/stop: `2`
- loop `834` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9662`
  best loop real test: acc `0.8983`, macro `0.8888`
  epochs run before resample/stop: `2`
- loop `835` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9662`
  best loop real test: acc `0.8984`, macro `0.8892`
  epochs run before resample/stop: `2`
- loop `836` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9662`
  best loop real test: acc `0.8988`, macro `0.8895`
  epochs run before resample/stop: `1`
- loop `837` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9662`
  best loop real test: acc `0.8983`, macro `0.8890`
  epochs run before resample/stop: `2`
- loop `838` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9662`
  best loop real test: acc `0.8983`, macro `0.8890`
  epochs run before resample/stop: `1`
- loop `839` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9662`
  best loop real test: acc `0.8986`, macro `0.8893`
  epochs run before resample/stop: `2`
- loop `840` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8983`, macro `0.8890`
  epochs run before resample/stop: `1`
- loop `841` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8983`, macro `0.8889`
  epochs run before resample/stop: `1`
- loop `842` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8984`, macro `0.8891`
  epochs run before resample/stop: `1`
- loop `843` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8983`, macro `0.8889`
  epochs run before resample/stop: `2`
- loop `844` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `2`
- loop `845` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8979`, macro `0.8886`
  epochs run before resample/stop: `2`
- loop `846` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8974`, macro `0.8880`
  epochs run before resample/stop: `2`
- loop `847` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `1`
- loop `848` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `2`
- loop `849` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `2`
- loop `850` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `1`
- loop `851` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `2`
- loop `852` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9662`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `2`
- loop `853` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9662`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `1`
- loop `854` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9662`
  best loop real test: acc `0.8968`, macro `0.8874`
  epochs run before resample/stop: `2`
- loop `855` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9662`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `856` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9662`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `2`
- loop `857` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9662`
  best loop real test: acc `0.8967`, macro `0.8872`
  epochs run before resample/stop: `2`
- loop `858` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9662`
  best loop real test: acc `0.8967`, macro `0.8872`
  epochs run before resample/stop: `1`
- loop `859` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `1`
- loop `860` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `1`
- loop `861` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `2`
- loop `862` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `1`
- loop `863` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `1`
- loop `864` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `2`
- loop `865` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8975`, macro `0.8880`
  epochs run before resample/stop: `2`
- loop `866` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `2`
- loop `867` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8970`, macro `0.8876`
  epochs run before resample/stop: `2`
- loop `868` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8968`, macro `0.8874`
  epochs run before resample/stop: `2`
- loop `869` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8967`, macro `0.8871`
  epochs run before resample/stop: `1`
- loop `870` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `1`
- loop `871` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `1`
- loop `872` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `1`
- loop `873` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8970`, macro `0.8874`
  epochs run before resample/stop: `1`
- loop `874` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `2`
- loop `875` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `2`
- loop `876` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `2`
- loop `877` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `1`
- loop `878` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8968`, macro `0.8873`
  epochs run before resample/stop: `2`
- loop `879` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `1`
- loop `880` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `2`
- loop `881` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `1`
- loop `882` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `1`
- loop `883` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `1`
- loop `884` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `1`
- loop `885` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `1`
- loop `886` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8975`, macro `0.8882`
  epochs run before resample/stop: `2`
- loop `887` accepted: `True`
  best synthetic val score: macro `0.9648`, acc `0.9659`
  best loop real test: acc `0.8975`, macro `0.8882`
  epochs run before resample/stop: `1`
- loop `888` accepted: `True`
  best synthetic val score: macro `0.9648`, acc `0.9659`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `2`
- loop `889` accepted: `True`
  best synthetic val score: macro `0.9648`, acc `0.9660`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `2`
- loop `890` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `2`
- loop `891` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `1`
- loop `892` accepted: `True`
  best synthetic val score: macro `0.9648`, acc `0.9659`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `1`
- loop `893` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `2`
- loop `894` accepted: `True`
  best synthetic val score: macro `0.9648`, acc `0.9659`
  best loop real test: acc `0.8975`, macro `0.8882`
  epochs run before resample/stop: `1`
- loop `895` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `2`
- loop `896` accepted: `True`
  best synthetic val score: macro `0.9648`, acc `0.9659`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `1`
- loop `897` accepted: `True`
  best synthetic val score: macro `0.9648`, acc `0.9659`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `1`
- loop `898` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `2`
- loop `899` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8981`, macro `0.8887`
  epochs run before resample/stop: `2`
- loop `900` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `2`
- loop `901` accepted: `True`
  best synthetic val score: macro `0.9648`, acc `0.9659`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `1`
- loop `902` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8981`, macro `0.8886`
  epochs run before resample/stop: `2`
- loop `903` accepted: `True`
  best synthetic val score: macro `0.9648`, acc `0.9660`
  best loop real test: acc `0.8981`, macro `0.8886`
  epochs run before resample/stop: `1`
- loop `904` accepted: `True`
  best synthetic val score: macro `0.9648`, acc `0.9659`
  best loop real test: acc `0.8983`, macro `0.8888`
  epochs run before resample/stop: `2`
- loop `905` accepted: `True`
  best synthetic val score: macro `0.9648`, acc `0.9659`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `1`
- loop `906` accepted: `True`
  best synthetic val score: macro `0.9648`, acc `0.9659`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `1`
- loop `907` accepted: `True`
  best synthetic val score: macro `0.9648`, acc `0.9659`
  best loop real test: acc `0.8967`, macro `0.8871`
  epochs run before resample/stop: `2`
- loop `908` accepted: `True`
  best synthetic val score: macro `0.9648`, acc `0.9659`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `1`
- loop `909` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8979`, macro `0.8885`
  epochs run before resample/stop: `2`
- loop `910` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8981`, macro `0.8888`
  epochs run before resample/stop: `2`
- loop `911` accepted: `True`
  best synthetic val score: macro `0.9648`, acc `0.9659`
  best loop real test: acc `0.8981`, macro `0.8888`
  epochs run before resample/stop: `1`
- loop `912` accepted: `True`
  best synthetic val score: macro `0.9647`, acc `0.9659`
  best loop real test: acc `0.8979`, macro `0.8885`
  epochs run before resample/stop: `1`
- loop `913` accepted: `True`
  best synthetic val score: macro `0.9647`, acc `0.9658`
  best loop real test: acc `0.8974`, macro `0.8878`
  epochs run before resample/stop: `2`
- loop `914` accepted: `True`
  best synthetic val score: macro `0.9647`, acc `0.9658`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `1`
- loop `915` accepted: `True`
  best synthetic val score: macro `0.9648`, acc `0.9659`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `2`
- loop `916` accepted: `True`
  best synthetic val score: macro `0.9648`, acc `0.9659`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `1`
- loop `917` accepted: `True`
  best synthetic val score: macro `0.9648`, acc `0.9658`
  best loop real test: acc `0.8970`, macro `0.8875`
  epochs run before resample/stop: `1`
- loop `918` accepted: `True`
  best synthetic val score: macro `0.9647`, acc `0.9659`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `2`
- loop `919` accepted: `True`
  best synthetic val score: macro `0.9647`, acc `0.9659`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `2`
- loop `920` accepted: `True`
  best synthetic val score: macro `0.9647`, acc `0.9659`
  best loop real test: acc `0.8979`, macro `0.8885`
  epochs run before resample/stop: `1`
- loop `921` accepted: `True`
  best synthetic val score: macro `0.9647`, acc `0.9659`
  best loop real test: acc `0.8974`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `922` accepted: `True`
  best synthetic val score: macro `0.9647`, acc `0.9658`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `2`
- loop `923` accepted: `True`
  best synthetic val score: macro `0.9647`, acc `0.9658`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `1`
- loop `924` accepted: `True`
  best synthetic val score: macro `0.9648`, acc `0.9659`
  best loop real test: acc `0.8983`, macro `0.8889`
  epochs run before resample/stop: `2`
- loop `925` accepted: `True`
  best synthetic val score: macro `0.9648`, acc `0.9659`
  best loop real test: acc `0.8977`, macro `0.8884`
  epochs run before resample/stop: `2`
- loop `926` accepted: `True`
  best synthetic val score: macro `0.9648`, acc `0.9659`
  best loop real test: acc `0.8979`, macro `0.8884`
  epochs run before resample/stop: `2`
- loop `927` accepted: `True`
  best synthetic val score: macro `0.9648`, acc `0.9659`
  best loop real test: acc `0.8975`, macro `0.8882`
  epochs run before resample/stop: `2`
- loop `928` accepted: `True`
  best synthetic val score: macro `0.9648`, acc `0.9659`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `1`
- loop `929` accepted: `True`
  best synthetic val score: macro `0.9648`, acc `0.9659`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `1`
- loop `930` accepted: `True`
  best synthetic val score: macro `0.9648`, acc `0.9659`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `1`
- loop `931` accepted: `True`
  best synthetic val score: macro `0.9648`, acc `0.9659`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `2`
- loop `932` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8986`, macro `0.8893`
  epochs run before resample/stop: `2`
- loop `933` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8984`, macro `0.8892`
  epochs run before resample/stop: `1`
- loop `934` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8981`, macro `0.8888`
  epochs run before resample/stop: `2`
- loop `935` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8979`, macro `0.8885`
  epochs run before resample/stop: `1`
- loop `936` accepted: `True`
  best synthetic val score: macro `0.9648`, acc `0.9660`
  best loop real test: acc `0.8984`, macro `0.8891`
  epochs run before resample/stop: `1`
- loop `937` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `2`
- loop `938` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `2`
- loop `939` accepted: `True`
  best synthetic val score: macro `0.9648`, acc `0.9660`
  best loop real test: acc `0.8981`, macro `0.8887`
  epochs run before resample/stop: `1`
- loop `940` accepted: `True`
  best synthetic val score: macro `0.9648`, acc `0.9660`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `1`
- loop `941` accepted: `True`
  best synthetic val score: macro `0.9648`, acc `0.9659`
  best loop real test: acc `0.8979`, macro `0.8885`
  epochs run before resample/stop: `1`
- loop `942` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `2`
- loop `943` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8979`, macro `0.8885`
  epochs run before resample/stop: `1`
- loop `944` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8981`, macro `0.8887`
  epochs run before resample/stop: `1`
- loop `945` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8981`, macro `0.8888`
  epochs run before resample/stop: `2`
- loop `946` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8984`, macro `0.8891`
  epochs run before resample/stop: `2`
- loop `947` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8984`, macro `0.8892`
  epochs run before resample/stop: `2`
- loop `948` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8984`, macro `0.8892`
  epochs run before resample/stop: `1`
- loop `949` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8986`, macro `0.8894`
  epochs run before resample/stop: `2`
- loop `950` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8986`, macro `0.8893`
  epochs run before resample/stop: `1`
- loop `951` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8984`, macro `0.8891`
  epochs run before resample/stop: `1`
- loop `952` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8986`, macro `0.8893`
  epochs run before resample/stop: `1`
- loop `953` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9661`
  best loop real test: acc `0.8986`, macro `0.8893`
  epochs run before resample/stop: `2`
- loop `954` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8984`, macro `0.8891`
  epochs run before resample/stop: `2`
- loop `955` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8984`, macro `0.8892`
  epochs run before resample/stop: `1`
- loop `956` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8983`, macro `0.8890`
  epochs run before resample/stop: `1`
- loop `957` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8983`, macro `0.8890`
  epochs run before resample/stop: `1`
- loop `958` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8983`, macro `0.8890`
  epochs run before resample/stop: `1`
- loop `959` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8983`, macro `0.8890`
  epochs run before resample/stop: `2`
- loop `960` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8983`, macro `0.8890`
  epochs run before resample/stop: `2`
- loop `961` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8981`, macro `0.8888`
  epochs run before resample/stop: `2`
- loop `962` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8979`, macro `0.8886`
  epochs run before resample/stop: `1`
- loop `963` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `1`
- loop `964` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8979`, macro `0.8885`
  epochs run before resample/stop: `1`
- loop `965` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `1`
- loop `966` accepted: `True`
  best synthetic val score: macro `0.9648`, acc `0.9659`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `1`
- loop `967` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `2`
- loop `968` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8983`, macro `0.8889`
  epochs run before resample/stop: `2`
- loop `969` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8981`, macro `0.8887`
  epochs run before resample/stop: `2`
- loop `970` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9660`
  best loop real test: acc `0.8975`, macro `0.8882`
  epochs run before resample/stop: `1`
- loop `971` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `2`
- loop `972` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `2`
- loop `973` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `1`
- loop `974` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9659`
  best loop real test: acc `0.8974`, macro `0.8880`
  epochs run before resample/stop: `1`
- loop `975` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `1`
- loop `976` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `1`
- loop `977` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `1`
- loop `978` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `2`
- loop `979` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8877`
  epochs run before resample/stop: `1`
- loop `980` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8979`, macro `0.8885`
  epochs run before resample/stop: `2`
- loop `981` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8983`, macro `0.8890`
  epochs run before resample/stop: `1`
- loop `982` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8983`, macro `0.8890`
  epochs run before resample/stop: `2`
- loop `983` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8983`, macro `0.8890`
  epochs run before resample/stop: `1`
- loop `984` accepted: `True`
  best synthetic val score: macro `0.9652`, acc `0.9661`
  best loop real test: acc `0.8981`, macro `0.8887`
  epochs run before resample/stop: `2`
- loop `985` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8981`, macro `0.8887`
  epochs run before resample/stop: `1`
- loop `986` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9661`
  best loop real test: acc `0.8983`, macro `0.8889`
  epochs run before resample/stop: `1`
- loop `987` accepted: `True`
  best synthetic val score: macro `0.9651`, acc `0.9660`
  best loop real test: acc `0.8979`, macro `0.8886`
  epochs run before resample/stop: `1`
- loop `988` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8979`, macro `0.8886`
  epochs run before resample/stop: `1`
- loop `989` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `2`
- loop `990` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8977`, macro `0.8884`
  epochs run before resample/stop: `2`
- loop `991` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8975`, macro `0.8882`
  epochs run before resample/stop: `2`
- loop `992` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `2`
- loop `993` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8974`, macro `0.8879`
  epochs run before resample/stop: `1`
- loop `994` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8974`, macro `0.8880`
  epochs run before resample/stop: `1`
- loop `995` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `2`
- loop `996` accepted: `True`
  best synthetic val score: macro `0.9648`, acc `0.9658`
  best loop real test: acc `0.8975`, macro `0.8881`
  epochs run before resample/stop: `1`
- loop `997` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `2`
- loop `998` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8977`, macro `0.8883`
  epochs run before resample/stop: `2`
- loop `999` accepted: `True`
  best synthetic val score: macro `0.9649`, acc `0.9659`
  best loop real test: acc `0.8983`, macro `0.8889`
  epochs run before resample/stop: `2`
- loop `1000` accepted: `True`
  best synthetic val score: macro `0.9650`, acc `0.9660`
  best loop real test: acc `0.8972`, macro `0.8878`
  epochs run before resample/stop: `2`

## Curves

![Training history](../experiment_runs/real_sim/cover_layer_branch_predictor_expert_principal_w1024_d6_acceptall_p1_1000_capped2_20260315/training_history.png)
