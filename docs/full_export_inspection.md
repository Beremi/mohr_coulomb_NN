# `constitutive_problem_3D_full.h5` Inspection

Inspected on 2026-03-13.

## Verdict

Yes. This export contains the key data we needed for the `U` / `B`-based route.

In particular, it contains:

- full displacement vector `U` for each captured constitutive call
- full sparse global strain-displacement operator `B`
- mesh geometry and element connectivity
- material identifiers per element
- per-integration-point strains `E`
- per-integration-point stresses `S`
- consistent tangents `DS` for a substantial subset of calls
- reduced constitutive parameters per integration point

Most importantly, the stored data are self-consistent:

- reconstructing `E` from `B @ U` matches exactly with zero numerical error in the checked sample call

So for FE-compatible synthetic strain generation, element-level kinematic modeling, and solver-trace analysis, this file is sufficient to proceed.

## File Summary

- file: `constitutive_problem_3D_full.h5`
- size: `14,397,885,516` bytes
- top-level groups:
  - `calls`
  - `problem`
- number of captured constitutive calls: `554`

Top-level file attributes confirm this is the full capture mode:

- `storage_mode = full`
- `stores_u = 1`
- `stores_ds = 1`
- `stores_problem_data = 1`
- `capture_target = CONSTITUTIVE_PROBLEM.constitutive_problem_3D`

## What Is In `calls`

Each call group, for example `calls/call_000001`, contains:

- `E`: `(202609, 6)` engineering strains
- `S`: `(202609, 6)` stresses
- `U`: `(1, 82815)` full displacement vector
- `c_bar`: `(202609, 1)`
- `sin_phi`: `(202609, 1)`
- `shear`: `(202609, 1)`
- `bulk`: `(202609, 1)`
- `lame`: `(202609, 1)`
- `DS`: `(202609, 36)` when tangent is stored

Call attributes include:

- `mode = stress_tangent` or `mode = stress`
- `has_DS = 1` or `0`
- `n_input_vectors = 202609`

Coverage:

- total calls: `554`
- calls with tangent `DS`: `157`
- calls with stress only: `397`

## What Is In `problem`

The `problem` group contains the FE data needed to reconstruct strains from displacements:

- `coord`: `(27605, 3)` node coordinates
- `elem`: `(18419, 10)` P2 tetra connectivity
- `material_identifier`: `(1, 18419)` material id per element
- `surf`: `(6325, 6)` surface connectivity
- `U_shape`: `(2, 1)` with value `(3, 27605)`
- `Q`: `(27605, 3)` nodal metadata / flags
- `B_coo`: sparse global `B` in COO form

`B_coo` contains:

- `row`: `(1, 17348874)`
- `col`: `(1, 17348874)`
- `val`: `(1, 17348874)`
- `shape`: `(2, 1)` with value `(1215654, 82815)`

This is exactly consistent with:

- `82815 = 3 * 27605` displacement DOFs
- `1215654 = 6 * 202609` strain rows

## Verified Consistency

I reconstructed the sparse matrix `B` from `problem/B_coo` and checked:

- `E_reconstructed = reshape(B @ U, (-1, 6))`

for `call_000001`.

Result:

- MAE: `0.0`
- RMSE: `0.0`
- max abs: `0.0`

So the file is not just storing related fields; it stores a fully consistent FE kinematic trace.

## Material Coverage

The mesh contains four material ids:

- `0`: `5124` elements
- `1`: `1357` elements
- `2`: `2603` elements
- `3`: `9335` elements

In the sampled inspected call, reduced constitutive parameters show four distinct strength states and three distinct elastic states, which is consistent with the heterogeneous material setup.

## What This Enables Now

This export is enough to support:

1. FE-compatible synthetic strain generation via `E = B U`
2. element-level displacement-distribution modeling
3. direct study of real deformation patterns instead of strain-only sampling
4. constitutive training/evaluation on physically compatible local states
5. tangent-supervised experiments on the subset of calls that provide `DS`

## Remaining Gaps

This export is very good, but a few items are still not explicitly stored:

- branch labels are not stored directly
  - we can recompute them from the exact constitutive operator
- raw geotechnical parameters / SRF / Davis-type metadata are not stored per point
  - reduced parameters are present, which is enough for learning
- explicit integration-point indexing metadata are not stored as a separate table
  - but the count `202609 = 18419 * 11` strongly indicates `11` quadrature points per element, consistent with the upstream formulation
- tangent `DS` is not stored for all calls
  - only `157 / 554`

None of these block the next stage.

## Bottom Line

For the workplan we discussed, this file contains what we need.

It is sufficient to start:

- `U`-space / `B`-pushforward synthetic augmentation
- element-compatible deformation modeling
- FE-structured constitutive training data generation

So the answer is: yes, this is the right kind of export, and it materially improves what we can do next.
