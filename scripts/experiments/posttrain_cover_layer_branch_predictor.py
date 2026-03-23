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
from torch.utils.data import DataLoader, TensorDataset


def _load_trainer_module():
    path = Path(__file__).with_name("train_cover_layer_strain_branch_predictor_synth_only.py")
    spec = importlib.util.spec_from_file_location("branch_trainer", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _score_improved(candidate: tuple[float, float, float, float], baseline: tuple[float, float, float, float]) -> bool:
    return candidate > baseline


def _expert_principal_loop_specs(elements_per_epoch: int, lr_scale: float) -> list[dict[str, object]]:
    return [
        {
            "name": "coverage_post",
            "base_lr": 2.0e-5 * lr_scale,
            "elements_per_epoch": elements_per_epoch,
            "recipe": [
                {"fraction": 0.50, "selection": "branch_balanced", "noise_scale": 0.20},
                {"fraction": 0.30, "selection": "boundary_smooth_right", "noise_scale": 0.05},
                {"fraction": 0.20, "selection": "tail", "noise_scale": 0.25},
            ],
        },
        {
            "name": "hard_post",
            "base_lr": 1.0e-5 * lr_scale,
            "elements_per_epoch": elements_per_epoch,
            "recipe": [
                {"fraction": 0.35, "selection": "smooth_edge", "noise_scale": 0.22},
                {"fraction": 0.40, "selection": "boundary_smooth_right", "noise_scale": 0.06},
                {"fraction": 0.25, "selection": "tail", "noise_scale": 0.30},
            ],
        },
    ]


def _write_posttrain_report(
    report_path: Path,
    *,
    artifact_dir: Path,
    summary: dict[str, object],
    base_summary: dict[str, object],
) -> None:
    rel = artifact_dir.relative_to(report_path.parent.parent)
    base_metrics = base_summary["metrics"]
    new_metrics = summary["metrics"]
    loop_results = summary["loop_results"]
    lines = [
        "# Cover Layer Branch Predictor Post-Train Report",
        "",
        "## Summary",
        "",
        f"- base checkpoint: `{summary['base_checkpoint']}`",
        f"- architecture: `w{summary['width']} d{summary['depth']}`",
        f"- feature set: `{summary['feature_set']}`",
        f"- generator: `{summary['generator_kind']}` / `{summary['recipe_mode']}`",
        f"- accepted post-train loops: `{summary['accepted_loops']}` out of `{summary['attempted_loops']}` attempted",
        "",
        "## Baseline vs Post-Train",
        "",
        f"- baseline synthetic test accuracy / macro recall: `{base_metrics['synthetic_test']['accuracy']:.4f}` / `{base_metrics['synthetic_test']['macro_recall']:.4f}`",
        f"- post-train synthetic test accuracy / macro recall: `{new_metrics['synthetic_test']['accuracy']:.4f}` / `{new_metrics['synthetic_test']['macro_recall']:.4f}`",
        f"- baseline real test accuracy / macro recall: `{base_metrics['real_test']['accuracy']:.4f}` / `{base_metrics['real_test']['macro_recall']:.4f}`",
        f"- post-train real test accuracy / macro recall: `{new_metrics['real_test']['accuracy']:.4f}` / `{new_metrics['real_test']['macro_recall']:.4f}`",
        "",
        "## Loop Results",
        "",
    ]
    for row in loop_results:
        lines.extend(
            [
                f"- loop `{row['loop_index']}` accepted: `{row['accepted']}`",
                f"  start score: macro `{row['start_score']['macro_recall']:.4f}`, acc `{row['start_score']['accuracy']:.4f}`",
                f"  end score: macro `{row['end_score']['macro_recall']:.4f}`, acc `{row['end_score']['accuracy']:.4f}`",
                f"  epochs added: `{row['epochs_added']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Curves",
            "",
            "![Training history](../" + str(rel / "training_history.png") + ")",
            "",
            "## Confusions",
            "",
            "![Confusions](../" + str(rel / "confusions.png") + ")",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    trainer = _load_trainer_module()

    parser = argparse.ArgumentParser(description="Post-train the current best cover-layer branch predictor until gains stop.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("experiment_runs/real_sim/cover_layer_branch_predictor_expert_principal_w1024_d6_long_20260315/best.pt"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("experiment_runs/real_sim/cover_layer_branch_predictor_expert_principal_w1024_d6_long_20260315/summary.json"),
    )
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
    parser.add_argument("--export", type=Path, default=Path("constitutive_problem_3D_full.h5"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiment_runs/real_sim/cover_layer_branch_predictor_expert_principal_w1024_d6_posttrain_20260315"),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("docs/cover_layer_branch_predictor_expert_principal_w1024_d6_posttrain.md"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-elements-per-call", type=int, default=128)
    parser.add_argument("--train-seed-calls", type=int, default=24)
    parser.add_argument("--eval-seed-calls", type=int, default=8)
    parser.add_argument("--synthetic-val-elements", type=int, default=16384)
    parser.add_argument("--synthetic-test-elements", type=int, default=16384)
    parser.add_argument("--batch-sizes", type=str, default="64,128,256,512,1024")
    parser.add_argument("--stage-max-epochs", type=int, default=30)
    parser.add_argument("--stage-patience", type=int, default=12)
    parser.add_argument("--plateau-patience", type=int, default=8)
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1.0e-6)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--lbfgs-epochs", type=int, default=2)
    parser.add_argument("--lbfgs-lr", type=float, default=0.15)
    parser.add_argument("--lbfgs-max-iter", type=int, default=20)
    parser.add_argument("--lbfgs-history-size", type=int, default=50)
    parser.add_argument("--elements-per-epoch", type=int, default=24576)
    parser.add_argument("--max-loops", type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_summary = json.loads(args.summary_json.read_text(encoding="utf-8"))
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    x_mean = np.asarray(ckpt["x_mean"], dtype=np.float32)
    x_std = np.asarray(ckpt["x_std"], dtype=np.float32)
    width = int(ckpt["width"])
    depth = int(ckpt["depth"])
    input_dim = int(ckpt["input_dim"])
    feature_set = str(ckpt["feature_set"])
    model_type = str(ckpt.get("model_type", "hierarchical"))
    generator_kind = str(ckpt.get("generator_kind", base_summary.get("generator_kind", "principal_hybrid")))
    recipe_mode = str(ckpt.get("recipe_mode", base_summary.get("recipe_mode", "expert_principal")))
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
        coords, disp, strain, branch, material = trainer.collect_blocks(
            args.export,
            call_names=call_names,
            max_elements_per_call=args.max_elements_per_call,
            seed=seed,
        )
        if generator_kind == "principal_hybrid":
            return trainer.fit_principal_hybrid_bank(strain, branch, material)
        bank = trainer.fit_seed_noise_bank(coords, disp, branch, material)
        bank["generator_kind"] = "element_seed"
        return bank

    train_seed_bank = build_seed_bank(train_seed_calls, args.seed + 1)
    eval_seed_bank = build_seed_bank(eval_seed_calls, args.seed + 2)

    def scale(x: np.ndarray) -> np.ndarray:
        return ((x - x_mean) / x_std).astype(np.float32)

    benchmark_recipe = [{"fraction": 1.0, "selection": "branch_balanced", "noise_scale": 0.20}]
    if generator_kind == "principal_hybrid" and recipe_mode == "expert_principal":
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
        "generator_kind": generator_kind,
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
    if generator_kind == "principal_hybrid":
        benchmark_summary["coverage"] = {
            "synthetic_val": trainer.summarize_branch_geometry(strain_syn_val, branch_syn_val, material_syn_val),
            "synthetic_test": trainer.summarize_branch_geometry(strain_syn_test, branch_syn_test, material_syn_test),
            "real_val": trainer.summarize_branch_geometry(strain_real_val, branch_real_val, material_real_val),
            "real_test": trainer.summarize_branch_geometry(strain_real_test, branch_real_test, material_real_test),
        }
    (args.output_dir / "benchmark_summary.json").write_text(json.dumps(benchmark_summary, indent=2), encoding="utf-8")
    trainer._plot_branch_frequencies(benchmark_summary["branch_frequencies"], args.output_dir / "benchmark_branch_frequencies.png")

    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    history: list[dict[str, float]] = []
    loop_results: list[dict[str, object]] = []
    start_time = time.time()

    current_metrics = trainer._evaluate_sets(model, eval_sets)
    best_score = trainer._score(current_metrics["synthetic_val"])
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_metrics = current_metrics
    accepted_loops = 0

    for loop_index in range(1, args.max_loops + 1):
        loop_start_score = best_score
        lr_scale = 0.5 ** (loop_index - 1)
        cycle_specs = _expert_principal_loop_specs(args.elements_per_epoch, lr_scale)
        loop_best_score = loop_start_score
        loop_best_state = copy.deepcopy(best_state)
        loop_best_metrics = best_metrics
        global_epoch_base = len(history)

        print(f"[posttrain-loop] loop={loop_index} lr_scale={lr_scale:.4f} start_score={loop_start_score}")

        for cycle_index, cycle in enumerate(cycle_specs, start=1):
            stage_lr = float(cycle["base_lr"])
            cycle_best_score = loop_best_score
            cycle_best_state = copy.deepcopy(loop_best_state)
            print(
                f"[cycle-start] loop={loop_index} cycle={cycle_index}/{len(cycle_specs)} "
                f"name={cycle['name']} base_lr={stage_lr:.2e} elements_per_epoch={cycle['elements_per_epoch']}"
            )

            for stage_index, batch_size in enumerate(batch_sizes, start=1):
                optimizer = torch.optim.AdamW(model.parameters(), lr=stage_lr, weight_decay=args.weight_decay)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="max",
                    factor=args.plateau_factor,
                    patience=args.plateau_patience,
                    min_lr=args.min_lr,
                )
                stage_best = cycle_best_score
                stage_no_improve = 0

                for local_epoch in range(args.stage_max_epochs):
                    strain_syn, branch_syn, material_syn = trainer._draw_training_recipe(
                        train_seed_bank,
                        recipe=cycle["recipe"],
                        element_count=int(cycle["elements_per_epoch"]),
                        seed=args.seed + 100000 * loop_index + 10000 * cycle_index + len(history) + local_epoch + 1,
                    )
                    x_train_np = trainer._build_point_features(strain_syn, material_syn, feature_set=feature_set)
                    y_train_np = trainer._flatten_pointwise_labels(branch_syn)
                    x_train = torch.from_numpy(scale(x_train_np))
                    y_train = torch.from_numpy(y_train_np)
                    class_weights = trainer._class_weights(y_train_np)
                    binary_weights = trainer._binary_class_weights(y_train_np)
                    plastic_weights = trainer._plastic_class_weights(y_train_np)
                    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)

                    model.train(True)
                    train_loss = 0.0
                    train_count = 0
                    for xb_cpu, yb_cpu in loader:
                        xb = xb_cpu.to(device)
                        yb = yb_cpu.to(device)
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
                        train_loss += float(loss.item()) * xb.shape[0]
                        train_count += xb.shape[0]
                    train_loss /= max(train_count, 1)

                    metrics_by_name = trainer._evaluate_sets(model, eval_sets)
                    score = trainer._score(metrics_by_name["synthetic_val"])
                    scheduler.step(score[0])
                    row = {
                        "global_epoch": len(history) + 1,
                        "loop_index": loop_index,
                        "cycle_index": cycle_index,
                        "cycle_name": cycle["name"],
                        "stage_index": stage_index,
                        "batch_size": batch_size,
                        "train_loss": train_loss,
                        "lr": optimizer.param_groups[0]["lr"],
                        "runtime_s": time.time() - start_time,
                    }
                    for split_name, split_metrics in metrics_by_name.items():
                        for key, value in split_metrics.items():
                            row[f"{split_name}_{key}"] = float(value)
                    history.append(row)

                    if score > cycle_best_score:
                        cycle_best_score = score
                        cycle_best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                        loop_best_score = score
                        loop_best_state = copy.deepcopy(cycle_best_state)
                        loop_best_metrics = metrics_by_name
                    if score > stage_best:
                        stage_best = score
                        stage_no_improve = 0
                    else:
                        stage_no_improve += 1

                    if len(history) == 1 or len(history) % 10 == 0:
                        print(
                            f"[epoch {len(history):04d}] loop={loop_index} cycle={cycle['name']} "
                            f"stage={stage_index}/{len(batch_sizes)} batch={batch_size} "
                            f"lr={optimizer.param_groups[0]['lr']:.2e} runtime={time.time()-start_time:.1f}s "
                            f"train_loss={train_loss:.4f} syn_val_acc={metrics_by_name['synthetic_val']['accuracy']:.4f} "
                            f"syn_val_macro={metrics_by_name['synthetic_val']['macro_recall']:.4f} "
                            f"real_test_acc={metrics_by_name['real_test']['accuracy']:.4f} "
                            f"real_test_macro={metrics_by_name['real_test']['macro_recall']:.4f}"
                        )
                    if stage_no_improve >= args.stage_patience:
                        break
                stage_lr = float(optimizer.param_groups[0]["lr"])

            model.load_state_dict(cycle_best_state)
            strain_cache, branch_cache, material_cache = trainer._draw_training_recipe(
                train_seed_bank,
                recipe=cycle["recipe"],
                element_count=4096,
                seed=args.seed + 990000 + 100 * loop_index + cycle_index,
            )
            x_cache_np = trainer._build_point_features(strain_cache, material_cache, feature_set=feature_set)
            y_cache_np = trainer._flatten_pointwise_labels(branch_cache)
            x_cache = torch.from_numpy(scale(x_cache_np)).to(device)
            y_cache = torch.from_numpy(y_cache_np).to(device)
            lbfgs_best, lbfgs_state = trainer._lbfgs_tail(
                model,
                x_train=x_cache,
                y_train=y_cache,
                eval_sets=eval_sets,
                model_type=model_type,
                class_weights=trainer._class_weights(y_cache_np),
                binary_weights=trainer._binary_class_weights(y_cache_np),
                plastic_weights=trainer._plastic_class_weights(y_cache_np),
                plastic_loss_weight=plastic_loss_weight,
                epochs=args.lbfgs_epochs,
                lr=args.lbfgs_lr,
                max_iter=args.lbfgs_max_iter,
                history_size=args.lbfgs_history_size,
                best_score=cycle_best_score,
            )
            if lbfgs_state is not None and lbfgs_best > cycle_best_score:
                cycle_best_score = lbfgs_best
                cycle_best_state = lbfgs_state
            model.load_state_dict(cycle_best_state)
            if cycle_best_score > loop_best_score:
                loop_best_score = cycle_best_score
                loop_best_state = copy.deepcopy(cycle_best_state)
                loop_best_metrics = trainer._evaluate_sets(model, eval_sets)

        improved = _score_improved(loop_best_score, loop_start_score)
        loop_result = {
            "loop_index": loop_index,
            "accepted": improved,
            "start_score": trainer._score_dict(loop_start_score),
            "end_score": trainer._score_dict(loop_best_score),
            "metrics": loop_best_metrics,
            "epochs_added": len(history) - global_epoch_base,
        }
        loop_results.append(loop_result)
        if improved:
            accepted_loops += 1
            best_score = loop_best_score
            best_state = copy.deepcopy(loop_best_state)
            best_metrics = loop_best_metrics
            model.load_state_dict(best_state)
            torch.save(
                {
                    **ckpt,
                    "state_dict": best_state,
                    "posttrain_loop": loop_index,
                },
                args.output_dir / f"loop_{loop_index:02d}_best.pt",
            )
        else:
            model.load_state_dict(best_state)
            break

    model.load_state_dict(best_state)
    final_metrics = trainer._evaluate_sets(model, eval_sets)
    confusions = {}
    for name, (x, y) in eval_sets.items():
        pred = trainer._predict_labels(model, x)
        confusions[name] = trainer._confusion_matrix(pred, y)

    trainer._plot_history(history, args.output_dir / "training_history.png")
    trainer._plot_confusions(confusions, args.output_dir / "confusions.png")

    checkpoint = {
        **ckpt,
        "state_dict": best_state,
        "posttrain_accepted_loops": accepted_loops,
    }
    torch.save(checkpoint, args.output_dir / "best.pt")

    summary = {
        "base_checkpoint": str(args.checkpoint),
        "generator_name": generator_kind,
        "feature_set": feature_set,
        "model_type": model_type,
        "generator_kind": generator_kind,
        "recipe_mode": recipe_mode,
        "plastic_loss_weight": plastic_loss_weight,
        "input_dim": input_dim,
        "width": width,
        "depth": depth,
        "accepted_loops": accepted_loops,
        "attempted_loops": len(loop_results),
        "batch_sizes": batch_sizes,
        "stage_max_epochs": args.stage_max_epochs,
        "stage_patience": args.stage_patience,
        "plateau_patience": args.plateau_patience,
        "plateau_factor": args.plateau_factor,
        "lbfgs_epochs": args.lbfgs_epochs,
        "score": trainer._score_dict(best_score),
        "metrics": final_metrics,
        "loop_results": loop_results,
        "checkpoint": str(args.output_dir / "best.pt"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    with (args.output_dir / "loop_results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["loop_index", "accepted", "start_macro", "end_macro", "start_acc", "end_acc", "epochs_added"])
        for row in loop_results:
            writer.writerow(
                [
                    row["loop_index"],
                    int(bool(row["accepted"])),
                    row["start_score"]["macro_recall"],
                    row["end_score"]["macro_recall"],
                    row["start_score"]["accuracy"],
                    row["end_score"]["accuracy"],
                    row["epochs_added"],
                ]
            )

    _write_posttrain_report(args.report_path, artifact_dir=args.output_dir, summary=summary, base_summary=base_summary)
    print(json.dumps({"summary": summary, "benchmark_summary": benchmark_summary}, indent=2))


if __name__ == "__main__":
    main()
