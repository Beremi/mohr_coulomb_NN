# Cover Layer B-Copula Detailed Statistics

This report focuses on the current best-screened B-copula mode and compares its pseudo-displacement and strain statistics against the real cover-layer data.

## Copula Reminder

The current model is a Gaussian copula, not a multivariate normal directly in `U`-space.
That means:

- the fitted latent normal variable `Z` has approximately zero mean
- dependence is carried by the latent covariance/correlation matrix of `Z`
- the physical `U` marginals are empirical, not Gaussian

Focused mode: `branch_raw`

## Screen Summary

| mode | total score | principal q95 logerr | principal q995 logerr | stress q95 logerr | stress q995 logerr | branch TV | synthetic test |
|---|---:|---:|---:|---:|---:|---:|---|
| `raw` | `3.1166` | `0.2122` | `0.4067` | `1.7045` | `0.0308` | `0.7625` | `experiment_runs/real_sim/cover_layer_b_copula_stats_20260312/fits/synthetic_test_raw.h5` |
| `pca` | `3.3726` | `0.2431` | `0.4077` | `1.9201` | `0.0298` | `0.7720` | `experiment_runs/real_sim/cover_layer_b_copula_stats_20260312/fits/synthetic_test_pca.h5` |
| `branch_raw` | `2.3355` | `0.6322` | `0.6603` | `0.9273` | `0.1042` | `0.0114` | `experiment_runs/real_sim/cover_layer_b_copula_stats_20260312/fits/synthetic_test_branch_raw.h5` |
| `branch_pca` | `2.3373` | `0.6385` | `0.6505` | `0.9337` | `0.1032` | `0.0114` | `experiment_runs/real_sim/cover_layer_b_copula_stats_20260312/fits/synthetic_test_branch_pca.h5` |
| `branch_raw_blend` | `3.2309` | `1.2448` | `0.9609` | `0.9113` | `0.1024` | `0.0114` | `experiment_runs/real_sim/cover_layer_b_copula_stats_20260312/fits/synthetic_test_branch_raw_blend.h5` |

## U Statistics

Real train pseudo-displacements `U/h`:

```text
[ 1.253104e-03  4.034369e-04  4.403918e-05 -7.718618e-04 -2.801727e-03 -1.272724e-03  1.610564e-03 -2.488557e-03 -1.044141e-03  3.880823e-03  9.918970e-04  1.025583e-03 -7.609856e-03 -4.834283e-03
 -3.751021e-03  9.580540e-04 -1.081803e-02 -1.531176e-03 -3.309411e-03 -3.585608e-03  1.116323e-03  3.274668e-03  1.258150e-03  8.943056e-04  5.144793e-03  6.885746e-03  1.291301e-03 -4.430876e-03
  1.498897e-02  3.227510e-03]
```

Generated pseudo-displacements `U/h`:

```text
[ 2.101494e-03 -1.944408e-03  2.308383e-03  1.085189e-03 -9.106344e-04  3.528744e-04 -4.097699e-04  5.064010e-03  2.395543e-03  3.993616e-03 -2.450234e-03 -2.296777e-03 -7.099691e-03 -2.818125e-03
 -5.771709e-04  3.402074e-04 -1.130396e-02  1.296056e-03 -1.050058e-02 -2.444217e-03  5.295573e-04  3.870441e-03  1.371234e-03  1.855954e-03  3.141546e-03  1.290582e-05 -2.635141e-03  7.520417e-03
  1.179836e-02  2.800176e-03]
```

Full real-train `U` covariance matrix: `experiment_runs/real_sim/cover_layer_b_copula_stats_20260312/branch_raw/real_train_u_cov.json`

Full generated `U` covariance matrix: `experiment_runs/real_sim/cover_layer_b_copula_stats_20260312/branch_raw/generated_u_cov.json`

![Real train U covariance](../experiment_runs/real_sim/cover_layer_b_copula_stats_20260312/branch_raw/real_train_u_cov.png)

![Generated U covariance](../experiment_runs/real_sim/cover_layer_b_copula_stats_20260312/branch_raw/generated_u_cov.png)

## E Statistics

Real train `E` mean:

```text
[ 0.709062  0.585498  0.156914 -3.201736  0.250708 -0.110982]
```

Real test `E` mean:

```text
[ 0.713017  0.618894  0.165195 -3.311651  0.15031  -0.112718]
```

Generated `E` mean:

```text
[ 0.540454  0.48913   0.308533 -0.323113  0.057023 -0.044174]
```

Real train `E` covariance:

```text
[[  2.253384  -0.065386  -0.60593   -2.499221   0.502334  -0.202655]
 [ -0.065386   7.715311  -1.04559  -19.507987   2.846096   0.117405]
 [ -0.60593   -1.04559    1.636341   0.722041  -0.213286  -0.289322]
 [ -2.499221 -19.507987   0.722041  58.894262  -8.491823   0.554913]
 [  0.502334   2.846096  -0.213286  -8.491823  18.496569  -2.035286]
 [ -0.202655   0.117405  -0.289322   0.554913  -2.035286   1.427871]]
```

Real test `E` covariance:

```text
[[ 2.183764e+00 -3.766468e-02 -6.280972e-01 -2.484674e+00  4.032218e-01 -1.253048e-01]
 [-3.766468e-02  8.162748e+00 -1.021775e+00 -2.089443e+01  1.819406e+00  1.919798e-01]
 [-6.280972e-01 -1.021775e+00  1.668501e+00  7.053960e-01 -3.880093e-02 -3.930753e-01]
 [-2.484674e+00 -2.089443e+01  7.053960e-01  6.286106e+01 -5.910433e+00  4.272057e-01]
 [ 4.032218e-01  1.819406e+00 -3.880093e-02 -5.910433e+00  1.983847e+01 -2.025470e+00]
 [-1.253048e-01  1.919798e-01 -3.930753e-01  4.272057e-01 -2.025470e+00  1.409993e+00]]
```

Generated `E` covariance:

```text
[[ 3.009219 -0.117169 -0.128923 -0.919968  0.314884  0.019199]
 [-0.117169  3.069864 -0.148112 -0.742137  0.014171 -0.267358]
 [-0.128923 -0.148112  1.379412  0.047806  0.106115 -0.146226]
 [-0.919968 -0.742137  0.047806  5.980458 -0.365778  0.473808]
 [ 0.314884  0.014171  0.106115 -0.365778  4.425645 -0.776952]
 [ 0.019199 -0.267358 -0.146226  0.473808 -0.776952  4.175779]]
```

![Real train E covariance](../experiment_runs/real_sim/cover_layer_b_copula_stats_20260312/branch_raw/real_train_e_cov.png)

![Real test E covariance](../experiment_runs/real_sim/cover_layer_b_copula_stats_20260312/branch_raw/real_test_e_cov.png)

![Generated E covariance](../experiment_runs/real_sim/cover_layer_b_copula_stats_20260312/branch_raw/generated_e_cov.png)

## Latent Gaussian Statistics

These are the actual Gaussian-copula statistics, in latent normal space.

### elastic

Latent mean:

```text
[ 7.484783e-18  6.092542e-18  4.390914e-18  3.275931e-17  2.760683e-17  2.854689e-17  5.925949e-18  2.943935e-17  1.394621e-17  3.361608e-17  2.732124e-17 -7.139698e-20 -8.603336e-18  2.647192e-17
 -1.580253e-17  1.848587e-17  2.510794e-17  2.698806e-17  1.260157e-17  2.579811e-17  5.664160e-17  5.378572e-18  1.765885e-17  2.610750e-17  2.194267e-17  1.451739e-17 -9.043617e-19  1.248257e-17
  2.972494e-17  2.170468e-17]
```

Full latent covariance matrix: `experiment_runs/real_sim/cover_layer_b_copula_stats_20260312/branch_raw/latent_cov_elastic.json`

![Latent covariance elastic](../experiment_runs/real_sim/cover_layer_b_copula_stats_20260312/branch_raw/latent_cov_elastic.png)

### smooth

Latent mean:

```text
[ 1.803675e-17  3.440317e-17  2.617281e-17  3.763605e-17 -8.257301e-18  2.292647e-17  3.367578e-19 -1.022397e-17  4.551618e-17  1.838697e-17  1.538310e-17  3.308308e-17  2.866482e-17 -8.486296e-19
  2.891402e-17  1.743058e-17 -2.143126e-17  1.347031e-20  1.306620e-18  3.663925e-18 -1.826574e-17  1.813104e-17  2.012464e-17  8.109127e-18  1.402259e-17  6.805201e-17 -8.957757e-18  6.013147e-17
  8.565771e-17  1.014314e-17]
```

Full latent covariance matrix: `experiment_runs/real_sim/cover_layer_b_copula_stats_20260312/branch_raw/latent_cov_smooth.json`

![Latent covariance smooth](../experiment_runs/real_sim/cover_layer_b_copula_stats_20260312/branch_raw/latent_cov_smooth.png)

### left_edge

Latent mean:

```text
[ 4.418044e-17  2.542046e-17  2.392598e-17  6.082085e-18  1.790497e-17  3.136244e-17  3.221745e-17  1.373768e-17  1.283956e-17  2.083644e-19 -1.632427e-17  1.195581e-17  2.461215e-17  2.986077e-17
  2.848126e-17  8.604013e-18  6.053345e-17  5.288145e-18  2.567193e-17  1.071640e-17  1.066969e-17  8.016282e-17  4.078194e-17  4.563899e-17 -9.484173e-19  2.339429e-17 -2.717359e-17  1.789060e-17
  2.907043e-17  3.135525e-17]
```

Full latent covariance matrix: `experiment_runs/real_sim/cover_layer_b_copula_stats_20260312/branch_raw/latent_cov_left_edge.json`

![Latent covariance left_edge](../experiment_runs/real_sim/cover_layer_b_copula_stats_20260312/branch_raw/latent_cov_left_edge.png)

### right_edge

Latent mean:

```text
[ 5.385478e-17 -1.371696e-17 -1.602676e-17  2.107185e-17 -2.918907e-17 -4.396722e-18  1.057645e-17 -1.430454e-17  1.801238e-17  4.052279e-19 -1.094115e-18 -4.111037e-17  2.026139e-18 -9.928083e-18
 -1.783003e-18  1.434507e-17 -4.153586e-17 -8.752922e-18 -3.647051e-18  3.525482e-18 -8.899817e-18  8.023512e-18  4.032017e-18  3.104045e-17  1.620911e-18 -1.934963e-17 -1.021174e-17  1.722218e-17
  1.013070e-17 -1.945094e-18]
```

Full latent covariance matrix: `experiment_runs/real_sim/cover_layer_b_copula_stats_20260312/branch_raw/latent_cov_right_edge.json`

![Latent covariance right_edge](../experiment_runs/real_sim/cover_layer_b_copula_stats_20260312/branch_raw/latent_cov_right_edge.png)

### apex

Latent mean:

```text
[ 5.072795e-18  4.853107e-18  1.387030e-17  9.087093e-18  1.268199e-17  1.485889e-17  2.070060e-17 -2.545385e-17  4.112159e-17  8.188369e-18  8.557844e-18  4.353816e-18 -1.698587e-17  2.083041e-17
  3.946894e-18 -2.465498e-17  3.356233e-17  2.650236e-17  1.471909e-17  1.461424e-17  1.385782e-17  1.909288e-17 -7.319603e-18  1.923268e-17  1.272193e-17  5.076789e-17  5.926582e-18 -1.228255e-17
 -1.549799e-17  9.416625e-18]
```

Full latent covariance matrix: `experiment_runs/real_sim/cover_layer_b_copula_stats_20260312/branch_raw/latent_cov_apex.json`

![Latent covariance apex](../experiment_runs/real_sim/cover_layer_b_copula_stats_20260312/branch_raw/latent_cov_apex.png)

