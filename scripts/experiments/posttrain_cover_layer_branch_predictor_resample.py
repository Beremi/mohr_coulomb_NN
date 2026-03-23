from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch


def _load_trainer_module():
    path = Path(__file__).with_name("train_cover_layer_strain_branch_predictor_synth_only.py")
    spec = importlib.util.spec_from_file_location("branch_trainer", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _draw_pointwise_recipe(
    trainer,
    seed_bank: dict[str, np.ndarray],
    *,
    recipe: list[dict[str, float | str]],
    point_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    strain_parts: list[np.ndarray] = []
    branch_parts: list[np.ndarray] = []
    material_parts: list[np.ndarray] = []
    assigned = 0
    for idx, item in enumerate(recipe):
        if idx == len(recipe) - 1:
            part_count = point_count - assigned
        else:
            part_count = int(round(point_count * float(item["fraction"])))
            part_count = min(part_count, point_count - assigned)
        if part_count <= 0:
            continue
        strain, branch, material, _valid = trainer.synthesize_from_principal_hybrid(
            seed_bank,
            sample_count=part_count,
            seed=seed + 1000 * (idx + 1),
            noise_scale=float(item["noise_scale"]),
            selection=str(item["selection"]),
        )
        strain_parts.append(strain)
        branch_parts.append(branch)
        material_parts.append(material)
        assigned += part_count
    return (
        np.concatenate(strain_parts, axis=0),
        np.concatenate(branch_parts, axis=0),
        np.concatenate(material_parts, axis=0),
    )


def _sample_replay_subset(
    replay_x_chunks: list[np.ndarray],
    replay_y_chunks: list[np.ndarray],
    *,
    limit: int,
    seed: int,
) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    if not replay_x_chunks:
        return None, None, 0
    lengths = np.asarray([chunk.shape[0] for chunk in replay_y_chunks], dtype=np.int64)
    total = int(lengths.sum())
    if total == 0:
        return None, None, 0
    if limit <= 0 or total <= limit:
        return (
            np.concatenate(replay_x_chunks, axis=0),
            np.concatenate(replay_y_chunks, axis=0),
            total,
        )
    rng = np.random.default_rng(seed)
    take = np.sort(rng.choice(total, size=limit, replace=False))
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    offset = 0
    take_pos = 0
    for x_chunk, y_chunk, chunk_len in zip(replay_x_chunks, replay_y_chunks, lengths, strict=False):
        chunk_start = offset
        chunk_end = offset + int(chunk_len)
        left = np.searchsorted(take, chunk_start, side="left", sorter=None)
        right = np.searchsorted(take, chunk_end, side="left", sorter=None)
        if right > left:
            local = take[left:right] - chunk_start
            x_parts.append(x_chunk[local])
            y_parts.append(y_chunk[local])
            take_pos = right
        offset = chunk_end
        if take_pos >= take.shape[0]:
            break
    return np.concatenate(x_parts, axis=0), np.concatenate(y_parts, axis=0), total


def _write_report(report_path: Path, *, artifact_dir: Path, summary: dict[str, object], base_summary: dict[str, object]) -> None:
    rel = artifact_dir.relative_to(report_path.parent.parent)
    base_metrics = base_summary["metrics"]
    final_metrics = summary["metrics"]
    lines = [
        "# Cover Layer Branch Predictor Resample Post-Train Report",
        "",
        "## Summary",
        "",
        f"- base checkpoint: `{summary['base_checkpoint']}`",
        f"- model: `w{summary['width']} d{summary['depth']}`",
        f"- feature set: `{summary['feature_set']}`",
        f"- training dataset per resample: `{summary['train_points']}` pointwise samples",
        f"- training batch size: `{summary['batch_size']}` pointwise samples",
        f"- replay mode: `{summary['replay_mode']}`",
        f"- replay cap per cycle: `{summary['replay_cap_points']}`",
        f"- final replay bank size: `{summary['final_replay_bank_size']}`",
        f"- optimizer: `{summary['optimizer']}`",
        f"- fixed LR: `{summary['learning_rate']:.1e}`",
        f"- per-loop patience before resampling: `{summary['patience']}`",
        f"- eval/checkpoint interval: every `{summary['eval_every_loops']}` loops",
        f"- acceptance mode: `{summary['acceptance_mode']}`",
        f"- attempted loops: `{summary['attempted_loops']}`",
        f"- accepted loops: `{summary['accepted_loops']}`",
        "",
        "## Baseline vs Final",
        "",
        f"- baseline synthetic test accuracy / macro recall: `{base_metrics['synthetic_test']['accuracy']:.4f}` / `{base_metrics['synthetic_test']['macro_recall']:.4f}`",
        f"- final synthetic test accuracy / macro recall: `{final_metrics['synthetic_test']['accuracy']:.4f}` / `{final_metrics['synthetic_test']['macro_recall']:.4f}`",
        f"- baseline real test accuracy / macro recall: `{base_metrics['real_test']['accuracy']:.4f}` / `{base_metrics['real_test']['macro_recall']:.4f}`",
        f"- final real test accuracy / macro recall: `{final_metrics['real_test']['accuracy']:.4f}` / `{final_metrics['real_test']['macro_recall']:.4f}`",
        "",
        "## Loop Results",
        "",
    ]
    for row in summary["loop_results"]:
        if not row.get("evaluated", True) and not row["accepted"]:
            continue
        lines.extend(
            [
                f"- loop `{row['loop_index']}` accepted: `{row['accepted']}`",
                f"  best synthetic val score: macro `{row['best_score']['macro_recall']:.4f}`, acc `{row['best_score']['accuracy']:.4f}`",
                f"  best loop real test: acc `{row['best_metrics']['real_test']['accuracy']:.4f}`, macro `{row['best_metrics']['real_test']['macro_recall']:.4f}`",
                f"  epochs run before resample/stop: `{row['epochs_run']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Curves",
            "",
            "![Training history](../" + str(rel / "training_history.png") + ")",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    trainer = _load_trainer_module()

    parser = argparse.ArgumentParser(description="Resample-on-patience post-train loops on top of the current best branch predictor.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("experiment_runs/real_sim/cover_layer_branch_predictor_expert_principal_w1024_d6_long_20260315/best.pt"),
    )
    parser.add_argument(
        "--base-summary-json",
        type=Path,
        default=Path("experiment_runs/real_sim/cover_layer_branch_predictor_expert_principal_w1024_d6_long_20260315/summary.json"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("experiment_runs/real_sim/cover_layer_branch_predictor_expert_principal_w1024_d6_resample_20260315/summary.json"),
    )
    parser.add_argument("--export", type=Path, default=Path("constitutive_problem_3D_full.h5"))
    parser.add_argument(
        "--split-json",
        type=Path,
        default=Path("experiment_runs/real_sim/cover_layer_branch_predictor_prep_20260314/call_splits.json"),
    )
    parser.add_argument(
        "--regime-json",
        type=Path,
        default=Path("experiment_runs/real_sim/cover_layer_branch_predictor_prep_20260314/call_regimes.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiment_runs/real_sim/cover_layer_branch_predictor_expert_principal_w1024_d6_resample_20260315"),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("docs/cover_layer_branch_predictor_expert_principal_w1024_d6_resample.md"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-elements-per-call", type=int, default=128)
    parser.add_argument("--train-seed-calls", type=int, default=24)
    parser.add_argument("--eval-seed-calls", type=int, default=8)
    parser.add_argument("--synthetic-val-elements", type=int, default=16384)
    parser.add_argument("--synthetic-test-elements", type=int, default=16384)
    parser.add_argument("--train-points", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--loops", type=int, default=10)
    parser.add_argument("--max-epochs-per-loop", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--optimizer", choices=("adamw", "lbfgs"), default="adamw")
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--lbfgs-max-iter", type=int, default=20)
    parser.add_argument("--lbfgs-history-size", type=int, default=50)
    parser.add_argument("--lbfgs-line-search", choices=("none", "strong_wolfe"), default="strong_wolfe")
    parser.add_argument("--replay-mispredictions", action="store_true")
    parser.add_argument("--replay-cap-points", type=int, default=0)
    parser.add_argument("--eval-every-loops", type=int, default=1)
    parser.add_argument("--accept-all-loops", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_summary = json.loads(args.base_summary_json.read_text(encoding="utf-8"))
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    x_mean = np.asarray(ckpt["x_mean"], dtype=np.float32)
    x_std = np.asarray(ckpt["x_std"], dtype=np.float32)
    width = int(ckpt["width"])
    depth = int(ckpt["depth"])
    input_dim = int(ckpt["input_dim"])
    feature_set = str(ckpt["feature_set"])
    model_type = str(ckpt.get("model_type", base_summary.get("model_type", "hierarchical")))
    plastic_loss_weight = float(ckpt.get("plastic_loss_weight", 1.0))

    if model_type == "flat":
        model = trainer.BranchMLP(in_dim=input_dim, width=width, depth=depth).to(device)
    else:
        model = trainer.HierarchicalBranchNet(in_dim=input_dim, width=width, depth=depth).to(device)
    model.load_state_dict(ckpt["state_dict"])

    splits = trainer.load_split_calls(args.split_json)
    regimes = trainer.load_call_regimes(args.regime_json)
    train_seed_calls, eval_seed_calls = trainer._split_seed_calls(
        splits["generator_fit"],
        regimes=regimes,
        train_count=args.train_seed_calls,
        eval_count=args.eval_seed_calls,
    )
    real_val_calls = trainer._spread_pick_exact(splits["real_val"], count=4, regimes=regimes)
    real_test_calls = trainer._spread_pick_exact(splits["real_test"], count=4, regimes=regimes)

    def build_seed_bank(call_names: list[str], seed: int) -> dict[str, np.ndarray]:
        _coords, _disp, strain, branch, material = trainer.collect_blocks(
            args.export,
            call_names=call_names,
            max_elements_per_call=args.max_elements_per_call,
            seed=seed,
        )
        return trainer.fit_principal_hybrid_bank(strain, branch, material)

    train_seed_bank = build_seed_bank(train_seed_calls, args.seed + 1)
    eval_seed_bank = build_seed_bank(eval_seed_calls, args.seed + 2)

    def scale(x: np.ndarray) -> np.ndarray:
        return ((x - x_mean) / x_std).astype(np.float32)

    benchmark_recipe = [
        {"fraction": 0.60, "selection": "branch_balanced", "noise_scale": 0.18},
        {"fraction": 0.25, "selection": "boundary_smooth_right", "noise_scale": 0.05},
        {"fraction": 0.15, "selection": "tail", "noise_scale": 0.25},
    ]
    strain_syn_val, branch_syn_val, material_syn_val = trainer._draw_training_recipe(
        eval_seed_bank,
        recipe=benchmark_recipe,
        element_count=args.synthetic_val_elements,
        seed=args.seed + 10,
    )
    strain_syn_test, branch_syn_test, material_syn_test = trainer._draw_training_recipe(
        eval_seed_bank,
        recipe=benchmark_recipe,
        element_count=args.synthetic_test_elements,
        seed=args.seed + 11,
    )
    x_syn_val_np = trainer._build_point_features(strain_syn_val, material_syn_val, feature_set=feature_set)
    y_syn_val_np = trainer._flatten_pointwise_labels(branch_syn_val)
    x_syn_test_np = trainer._build_point_features(strain_syn_test, material_syn_test, feature_set=feature_set)
    y_syn_test_np = trainer._flatten_pointwise_labels(branch_syn_test)
    x_syn_val = torch.from_numpy(scale(x_syn_val_np)).to(device)
    y_syn_val = torch.from_numpy(y_syn_val_np).to(device)
    x_syn_test = torch.from_numpy(scale(x_syn_test_np)).to(device)
    y_syn_test = torch.from_numpy(y_syn_test_np).to(device)

    _, _, strain_real_val, branch_real_val, material_real_val = trainer.collect_blocks(
        args.export,
        call_names=real_val_calls,
        max_elements_per_call=args.max_elements_per_call,
        seed=args.seed + 20,
    )
    _, _, strain_real_test, branch_real_test, material_real_test = trainer.collect_blocks(
        args.export,
        call_names=real_test_calls,
        max_elements_per_call=args.max_elements_per_call,
        seed=args.seed + 21,
    )
    x_real_val_np = trainer._build_point_features(strain_real_val, material_real_val, feature_set=feature_set)
    y_real_val_np = trainer._flatten_pointwise_labels(branch_real_val)
    x_real_test_np = trainer._build_point_features(strain_real_test, material_real_test, feature_set=feature_set)
    y_real_test_np = trainer._flatten_pointwise_labels(branch_real_test)
    x_real_val = torch.from_numpy(scale(x_real_val_np)).to(device)
    y_real_val = torch.from_numpy(y_real_val_np).to(device)
    x_real_test = torch.from_numpy(scale(x_real_test_np)).to(device)
    y_real_test = torch.from_numpy(y_real_test_np).to(device)

    eval_sets = {
        "synthetic_val": (x_syn_val, y_syn_val),
        "synthetic_test": (x_syn_test, y_syn_test),
        "real_val": (x_real_val, y_real_val),
        "real_test": (x_real_test, y_real_test),
    }

    benchmark_summary = {
        "train_points": args.train_points,
        "batch_size": args.batch_size,
        "replay_mode": "misprediction_bank" if args.replay_mispredictions else "none",
        "replay_cap_points": args.replay_cap_points,
        "loops": args.loops,
        "patience": args.patience,
        "eval_every_loops": args.eval_every_loops,
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "train_seed_calls": train_seed_calls,
        "eval_seed_calls": eval_seed_calls,
        "real_val_calls": real_val_calls,
        "real_test_calls": real_test_calls,
        "synthetic_val_elements": args.synthetic_val_elements,
        "synthetic_test_elements": args.synthetic_test_elements,
        "branch_frequencies": {
            "synthetic_val": trainer._branch_frequencies(y_syn_val_np),
            "synthetic_test": trainer._branch_frequencies(y_syn_test_np),
            "real_val": trainer._branch_frequencies(y_real_val_np),
            "real_test": trainer._branch_frequencies(y_real_test_np),
        },
    }
    (args.output_dir / "benchmark_summary.json").write_text(json.dumps(benchmark_summary, indent=2), encoding="utf-8")
    trainer._plot_branch_frequencies(benchmark_summary["branch_frequencies"], args.output_dir / "benchmark_branch_frequencies.png")

    loop_recipe = [
        {"fraction": 0.35, "selection": "branch_balanced", "noise_scale": 0.20},
        {"fraction": 0.30, "selection": "boundary_smooth_right", "noise_scale": 0.05},
        {"fraction": 0.20, "selection": "smooth_edge", "noise_scale": 0.22},
        {"fraction": 0.15, "selection": "tail", "noise_scale": 0.25},
    ]

    current_metrics = trainer._evaluate_sets(model, eval_sets)
    current_score = trainer._score(current_metrics["synthetic_val"])
    current_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    history: list[dict[str, float]] = []
    loop_results: list[dict[str, object]] = []
    accepted_loops = 0
    start_time = time.time()
    best_seen_score = current_score
    best_seen_state = copy.deepcopy(current_state)
    best_seen_metrics = current_metrics
    replay_x_chunks: list[np.ndarray] = []
    replay_y_chunks: list[np.ndarray] = []
    pending_accept_start_score = current_score
    pending_accept_start_state = copy.deepcopy(current_state)
    pending_accept_start_metrics = current_metrics
    pending_block_has_eval = False

    for loop_index in range(1, args.loops + 1):
        strain_batch, branch_batch, material_batch = _draw_pointwise_recipe(
            trainer,
            train_seed_bank,
            recipe=loop_recipe,
            point_count=args.train_points,
            seed=args.seed + 10000 * loop_index,
        )
        x_fresh_np = trainer._build_point_features(strain_batch, material_batch, feature_set=feature_set)
        y_fresh_np = trainer._flatten_pointwise_labels(branch_batch)
        x_fresh_scaled = scale(x_fresh_np)
        replay_active_x, replay_active_y, replay_total = _sample_replay_subset(
            replay_x_chunks,
            replay_y_chunks,
            limit=args.replay_cap_points,
            seed=args.seed + 500000 * loop_index,
        )
        if replay_active_x is not None:
            x_train_scaled = np.concatenate([x_fresh_scaled, replay_active_x], axis=0)
            y_train_np = np.concatenate([y_fresh_np, replay_active_y], axis=0)
        else:
            x_train_scaled = x_fresh_scaled
            y_train_np = y_fresh_np
        x_train = torch.from_numpy(x_train_scaled).to(device)
        y_train = torch.from_numpy(y_train_np).to(device)
        x_fresh = torch.from_numpy(x_fresh_scaled).to(device)
        y_fresh = torch.from_numpy(y_fresh_np).to(device)

        class_weights = trainer._class_weights(y_train_np)
        binary_weights = trainer._binary_class_weights(y_train_np)
        plastic_weights = trainer._plastic_class_weights(y_train_np)
        if args.optimizer == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.LBFGS(
                model.parameters(),
                lr=args.learning_rate,
                max_iter=args.lbfgs_max_iter,
                history_size=args.lbfgs_history_size,
                line_search_fn=None if args.lbfgs_line_search == "none" else args.lbfgs_line_search,
            )

        loop_best_score = current_score
        loop_best_state = copy.deepcopy(current_state)
        loop_best_metrics = current_metrics
        no_improve = 0

        print(
            f"[resample-loop] loop={loop_index}/{args.loops} train_points={args.train_points} "
            f"batch={args.batch_size} replay_active={0 if replay_active_y is None else replay_active_y.shape[0]} "
            f"replay_total={replay_total} optimizer={args.optimizer} lr={args.learning_rate:.2e} patience={args.patience}"
        )

        for epoch in range(1, args.max_epochs_per_loop + 1):
            model.train(True)
            if args.optimizer == "adamw":
                perm = torch.randperm(x_train.shape[0], device=device)
                total_loss = 0.0
                total_seen = 0
                for start in range(0, x_train.shape[0], args.batch_size):
                    idx = perm[start : start + args.batch_size]
                    xb = x_train[idx]
                    yb = y_train[idx]
                    optimizer.zero_grad(set_to_none=True)
                    loss = trainer._compute_loss(
                        model,
                        xb,
                        yb,
                        model_type=model_type,
                        class_weights=class_weights,
                        binary_weights=binary_weights,
                        plastic_weights=plastic_weights,
                        plastic_loss_weight=plastic_loss_weight,
                    )
                    loss.backward()
                    optimizer.step()
                    batch_n = int(yb.shape[0])
                    total_loss += float(loss.item()) * batch_n
                    total_seen += batch_n
                mean_loss = total_loss / max(total_seen, 1)
            else:
                def closure():
                    optimizer.zero_grad(set_to_none=True)
                    loss = trainer._compute_loss(
                        model,
                        x_train,
                        y_train,
                        model_type=model_type,
                        class_weights=class_weights,
                        binary_weights=binary_weights,
                        plastic_weights=plastic_weights,
                        plastic_loss_weight=plastic_loss_weight,
                    )
                    loss.backward()
                    return loss

                loss = optimizer.step(closure)
                mean_loss = float(loss.item())
            should_eval = (loop_index % args.eval_every_loops == 0) or (loop_index == args.loops)
            if should_eval:
                metrics_by_name = trainer._evaluate_sets(model, eval_sets)
                score = trainer._score(metrics_by_name["synthetic_val"])
                pending_block_has_eval = True
            else:
                metrics_by_name = current_metrics
                score = current_score

            row = {
                "global_epoch": len(history) + 1,
                "loop_index": loop_index,
                "epoch_in_loop": epoch,
                "train_loss": float(mean_loss),
                "lr": args.learning_rate,
                "replay_total": float(replay_total),
                "replay_active": float(0 if replay_active_y is None else replay_active_y.shape[0]),
                "runtime_s": time.time() - start_time,
            }
            for split_name, split_metrics in metrics_by_name.items():
                for key, value in split_metrics.items():
                    row[f"{split_name}_{key}"] = float(value)
            history.append(row)

            if score > loop_best_score:
                loop_best_score = score
                loop_best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                loop_best_metrics = metrics_by_name
                no_improve = 0
            else:
                no_improve += 1

            if len(history) == 1 or len(history) % 5 == 0:
                print(
                    f"[epoch {len(history):04d}] loop={loop_index} train_loss={mean_loss:.4f} "
                    f"syn_val_acc={metrics_by_name['synthetic_val']['accuracy']:.4f} "
                    f"syn_val_macro={metrics_by_name['synthetic_val']['macro_recall']:.4f} "
                    f"real_test_acc={metrics_by_name['real_test']['accuracy']:.4f} "
                    f"real_test_macro={metrics_by_name['real_test']['macro_recall']:.4f} "
                    f"no_improve={no_improve}"
                )

            if no_improve >= args.patience:
                break

        if args.replay_mispredictions:
            model.eval()
            with torch.no_grad():
                fresh_pred = trainer._predict_labels(model, x_fresh)
            wrong_mask = (fresh_pred != y_fresh).detach().cpu().numpy().astype(bool)
            wrong_count = int(wrong_mask.sum())
            if wrong_count > 0:
                replay_x_chunks.append(x_fresh_scaled[wrong_mask].astype(np.float32, copy=False))
                replay_y_chunks.append(y_fresh_np[wrong_mask].astype(np.int64, copy=False))
        else:
            wrong_count = 0

        should_accept_eval = (loop_index % args.eval_every_loops == 0) or (loop_index == args.loops)
        if args.accept_all_loops and should_accept_eval:
            final_loop_metrics = trainer._evaluate_sets(model, eval_sets)
            final_loop_score = trainer._score(final_loop_metrics["synthetic_val"])
            accepted = True
            current_score = final_loop_score
            current_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            current_metrics = final_loop_metrics
            if final_loop_score > best_seen_score:
                best_seen_score = final_loop_score
                best_seen_state = copy.deepcopy(current_state)
                best_seen_metrics = final_loop_metrics
        elif (not args.accept_all_loops) and should_accept_eval:
            accepted = pending_block_has_eval and (loop_best_score > pending_accept_start_score)
        else:
            accepted = False
        loop_results.append(
            {
                "loop_index": loop_index,
                "accepted": accepted,
                "best_score": trainer._score_dict(loop_best_score if not args.accept_all_loops else current_score),
                "best_metrics": loop_best_metrics if not args.accept_all_loops else current_metrics,
                "epochs_run": epoch,
                "fresh_wrong_count": wrong_count,
                "replay_total_after": int(sum(chunk.shape[0] for chunk in replay_y_chunks)),
                "evaluated": should_accept_eval,
            }
        )
        if accepted and not args.accept_all_loops:
            accepted_loops += 1
            current_score = loop_best_score
            current_state = copy.deepcopy(loop_best_state)
            current_metrics = loop_best_metrics
            model.load_state_dict(current_state)
            torch.save({**ckpt, "state_dict": current_state, "accepted_loop": loop_index}, args.output_dir / f"loop_{loop_index:02d}_accepted.pt")
            pending_accept_start_score = current_score
            pending_accept_start_state = copy.deepcopy(current_state)
            pending_accept_start_metrics = current_metrics
            pending_block_has_eval = False
        elif accepted and args.accept_all_loops:
            accepted_loops += 1
            model.load_state_dict(current_state)
            if current_score == best_seen_score:
                torch.save({**ckpt, "state_dict": best_seen_state, "accepted_loop": loop_index}, args.output_dir / f"loop_{loop_index:04d}_best_seen.pt")
            pending_accept_start_score = current_score
            pending_accept_start_state = copy.deepcopy(current_state)
            pending_accept_start_metrics = current_metrics
            pending_block_has_eval = False
        elif should_accept_eval:
            model.load_state_dict(pending_accept_start_state)
            current_score = pending_accept_start_score
            current_state = copy.deepcopy(pending_accept_start_state)
            current_metrics = pending_accept_start_metrics
            pending_block_has_eval = False
        else:
            pass

    model.load_state_dict(current_state)
    final_metrics = trainer._evaluate_sets(model, eval_sets)
    trainer._plot_history(history, args.output_dir / "training_history.png")

    checkpoint = {
        **ckpt,
        "state_dict": current_state,
        "train_points": args.train_points,
        "batch_size": args.batch_size,
        "accepted_loops": accepted_loops,
        "resample_patience": args.patience,
        "resample_learning_rate": args.learning_rate,
        "accept_all_loops": args.accept_all_loops,
        "optimizer": args.optimizer,
        "replay_mispredictions": args.replay_mispredictions,
        "replay_cap_points": args.replay_cap_points,
    }
    torch.save(checkpoint, args.output_dir / "best.pt")
    if args.accept_all_loops:
        torch.save({**ckpt, "state_dict": best_seen_state, "acceptance_mode": "best_seen_within_run"}, args.output_dir / "best_seen.pt")

    summary = {
        "base_checkpoint": str(args.checkpoint),
        "feature_set": feature_set,
        "model_type": model_type,
        "generator_kind": "principal_hybrid",
        "recipe_mode": "resample_posttrain",
        "plastic_loss_weight": plastic_loss_weight,
        "input_dim": input_dim,
        "width": width,
        "depth": depth,
        "train_points": args.train_points,
        "batch_size": args.batch_size,
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "patience": args.patience,
        "eval_every_loops": args.eval_every_loops,
        "acceptance_mode": "accept_all_loops" if args.accept_all_loops else "improve_only",
        "replay_mode": "misprediction_bank" if args.replay_mispredictions else "none",
        "replay_cap_points": args.replay_cap_points,
        "final_replay_bank_size": int(sum(chunk.shape[0] for chunk in replay_y_chunks)),
        "loops": args.loops,
        "accepted_loops": accepted_loops,
        "attempted_loops": args.loops,
        "score": trainer._score_dict(current_score),
        "metrics": final_metrics,
        "best_seen_score": trainer._score_dict(best_seen_score),
        "best_seen_metrics": best_seen_metrics,
        "loop_results": loop_results,
        "checkpoint": str(args.output_dir / "best.pt"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    with (args.output_dir / "loop_results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "loop_index",
                "evaluated",
                "accepted",
                "syn_val_macro",
                "syn_val_acc",
                "real_test_macro",
                "real_test_acc",
                "epochs_run",
                "fresh_wrong_count",
                "replay_total_after",
            ]
        )
        for row in loop_results:
            writer.writerow(
                [
                    row["loop_index"],
                    int(bool(row.get("evaluated", True))),
                    int(bool(row["accepted"])),
                    row["best_score"]["macro_recall"],
                    row["best_score"]["accuracy"],
                    row["best_metrics"]["real_test"]["macro_recall"],
                    row["best_metrics"]["real_test"]["accuracy"],
                    row["epochs_run"],
                    row["fresh_wrong_count"],
                    row["replay_total_after"],
                ]
            )

    _write_report(args.report_path, artifact_dir=args.output_dir, summary=summary, base_summary=base_summary)
    print(json.dumps({"summary": summary, "benchmark_summary": benchmark_summary}, indent=2))


if __name__ == "__main__":
    main()
