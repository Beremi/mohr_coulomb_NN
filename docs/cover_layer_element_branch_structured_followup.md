# Cover Layer Direct Element Branch Predictor: Structured Follow-Up

## Summary

This follow-up does two things:

1. separates the old Gate B failure into `B1/B2/B3` on the flat direct MLP
2. replaces the flat MLP with a structured raw-element model that also predicts auxiliary strain

The input/output contract is unchanged:

- input: canonicalized `coords(10x3) + disp(10x3)`
- output: `11 x 5` branch logits

## B1 / B2 / B3 Diagnosis with the Flat Direct MLP

`B1` = same train seed pool, disjoint synthetic samples.
`B2` = disjoint synthetic seed-call validation.
`B3` = expanded train seed-call pool (`48`) against disjoint eval seed calls (`8`).

- `b1_same_seed_pool`: val acc `0.3586`, val macro `0.3783`, val pattern `0.0063`
- `b2_disjoint_seed_calls`: val acc `0.3390`, val macro `0.3371`, val pattern `0.0005`
- `b3_expanded_train_calls`: val acc `0.3347`, val macro `0.3351`, val pattern `0.0059`

![B-diagnostic pattern accuracy](../experiment_runs/real_sim/cover_layer_element_branch_structured_followup_20260314/b_diagnostic_pattern_accuracy.png)
![B-diagnostic point accuracy](../experiment_runs/real_sim/cover_layer_element_branch_structured_followup_20260314/b_diagnostic_point_accuracy.png)

Interpretation:

- if `B1` is already poor, the flat direct MLP does not generalize within the synthetic domain itself
- if `B1` is good but `B2/B3` collapse, the main problem is cross-seed-call generalization

## Structured Raw-Element Model

Architecture:

- node input: `10 x 6` canonicalized `(x,y,z,u,v,w)`
- node encoder with role embeddings for the `10` P2 nodes
- self-attention across nodes
- `11` learned integration-point queries with cross-attention into node tokens
- branch head: `11 x 5`
- auxiliary strain head: `11 x 6`

The strain head is supervised with exact synthetic strain and used only as an auxiliary training target.

## Structured Gate A

- synthetic val acc: `0.9996`
- synthetic val macro recall: `0.9998`
- synthetic val pattern accuracy: `0.9961`
- synthetic val strain MAE: `0.3176`

![Structured Gate A history](../experiment_runs/real_sim/cover_layer_element_branch_structured_followup_20260314/structured_gate_a/history.png)

## Structured Gate B

- synthetic val acc: `0.1835`
- synthetic val macro recall: `0.2758`
- synthetic val pattern accuracy: `0.0005`
- synthetic val strain MAE: `1.4222`

- real test acc: `0.3590`
- real test macro recall: `0.3482`
- real test pattern accuracy: `0.1172`

![Structured Gate B history](../experiment_runs/real_sim/cover_layer_element_branch_structured_followup_20260314/structured_gate_b/history.png)

## Conclusion

This follow-up is trying to answer whether the next move is still training recipe or now architecture.

- If the structured model improves Gate B materially, then the flat MLP was the main bottleneck.
- If the structured model is still poor on synthetic Gate B, then the synthetic domain itself is too diverse for the current direct formulation, and the next move should be to narrow or factor the task.
