from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def _load_trainer_module():
    script_path = Path(__file__).with_name("train_cover_layer_strain_branch_predictor_synth_only.py")
    spec = importlib.util.spec_from_file_location("cover_branch_trainer", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load trainer module from {script_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inflate_tensor(
    old_tensor: torch.Tensor,
    new_shape: torch.Size,
    *,
    generator: torch.Generator,
    noise_scale: float,
) -> torch.Tensor:
    new_tensor = torch.empty(new_shape, dtype=old_tensor.dtype)
    new_tensor.normal_(mean=0.0, std=noise_scale, generator=generator)
    slices = tuple(slice(0, dim) for dim in old_tensor.shape)
    new_tensor[slices] = old_tensor
    return new_tensor


def _inflate_checkpoint(
    ckpt: dict[str, object],
    trainer,
    *,
    new_width: int,
    noise_scale: float,
    seed: int,
) -> dict[str, object]:
    old_width = int(ckpt["width"])
    if new_width < old_width:
        raise ValueError(f"Requested width {new_width} is smaller than checkpoint width {old_width}.")
    if new_width == old_width:
        return copy.deepcopy(ckpt)

    model_type = str(ckpt.get("model_type", "hierarchical"))
    input_dim = int(ckpt["input_dim"])
    depth = int(ckpt["depth"])
    if model_type == "flat":
        new_model = trainer.BranchMLP(in_dim=input_dim, width=new_width, depth=depth)
    elif model_type == "hierarchical":
        new_model = trainer.HierarchicalBranchNet(in_dim=input_dim, width=new_width, depth=depth)
    else:
        raise ValueError(f"Unsupported model_type {model_type!r}.")

    old_state = ckpt["state_dict"]
    new_state: dict[str, torch.Tensor] = {}
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    for name, param in new_model.state_dict().items():
        old_param = old_state[name]
        if tuple(old_param.shape) == tuple(param.shape):
            new_state[name] = old_param.detach().cpu().clone()
        else:
            new_state[name] = _inflate_tensor(
                old_param.detach().cpu(),
                param.shape,
                generator=generator,
                noise_scale=noise_scale,
            )

    inflated = copy.deepcopy(ckpt)
    inflated["width"] = new_width
    inflated["state_dict"] = new_state
    return inflated


def _prediction_change_rate(trainer, model_old, model_new, x: torch.Tensor) -> float:
    with torch.no_grad():
        pred_old = trainer._predict_labels(model_old, x)
        pred_new = trainer._predict_labels(model_new, x)
    return float((pred_old != pred_new).float().mean().item())


def _evaluate_phase(trainer, model: torch.nn.Module, eval_sets: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> dict[str, dict[str, float]]:
    return trainer._evaluate_sets(model, eval_sets)


def _move_eval_sets(
    eval_sets: dict[str, tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    return {
        name: (x.to(device), y.to(device))
        for name, (x, y) in eval_sets.items()
    }


def _score_tuple(trainer, metrics_by_name: dict[str, dict[str, float]]) -> tuple[float, float, float, float]:
    return trainer._score(metrics_by_name["synthetic_val"])


def _save_checkpoint(
    path: Path,
    *,
    base_ckpt: dict[str, object],
    model: torch.nn.Module,
    width: int,
    phase_name: str,
) -> None:
    ckpt = copy.deepcopy(base_ckpt)
    ckpt["width"] = width
    ckpt["state_dict"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    ckpt["phase_name"] = phase_name
    torch.save(ckpt, path)


def _run_adam_phase(
    *,
    trainer,
    model: torch.nn.Module,
    train_seed_bank: dict[str, np.ndarray],
    loop_recipe: list[dict[str, float | str]],
    feature_set: str,
    scale,
    eval_sets: dict[str, tuple[torch.Tensor, torch.Tensor]],
    train_points: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    loops: int,
    seed: int,
    device: torch.device,
    history_rows: list[dict[str, float | str]],
    global_step_start: int,
) -> tuple[int, dict[str, dict[str, float]]]:
    global_step = global_step_start
    start = time.time()
    for loop_index in range(1, loops + 1):
        strain_batch, branch_batch, material_batch = trainer._draw_training_recipe(
            train_seed_bank,
            recipe=loop_recipe,
            element_count=train_points,
            seed=seed + 10000 * loop_index,
        )
        x_train_np = trainer._build_point_features(strain_batch, material_batch, feature_set=feature_set)
        y_train_np = trainer._flatten_pointwise_labels(branch_batch)
        x_train = torch.from_numpy(scale(x_train_np))
        y_train = torch.from_numpy(y_train_np)
        loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)

        class_weights = trainer._class_weights(y_train_np)
        binary_weights = trainer._binary_class_weights(y_train_np)
        plastic_weights = trainer._plastic_class_weights(y_train_np)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

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
                model_type="hierarchical",
                class_weights=class_weights,
                binary_weights=binary_weights,
                plastic_weights=plastic_weights,
                plastic_loss_weight=1.0,
            )
            loss.backward()
            optimizer.step()
            n = int(yb.shape[0])
            train_loss += float(loss.item()) * n
            train_count += n
        metrics_by_name = _evaluate_phase(trainer, model, eval_sets)
        global_step += 1
        row: dict[str, float | str] = {
            "phase": "adam",
            "global_epoch": float(global_step),
            "loop_index": float(loop_index),
            "step_in_loop": 1.0,
            "train_loss": train_loss / max(train_count, 1),
            "lr": learning_rate,
            "runtime_s": time.time() - start,
        }
        for split_name, split_metrics in metrics_by_name.items():
            for key, value in split_metrics.items():
                row[f"{split_name}_{key}"] = float(value)
        history_rows.append(row)
        print(
            f"[adam] loop={loop_index}/{loops} loss={row['train_loss']:.6f} "
            f"syn_val_acc={metrics_by_name['synthetic_val']['accuracy']:.4f} "
            f"syn_val_macro={metrics_by_name['synthetic_val']['macro_recall']:.4f} "
            f"real_test_acc={metrics_by_name['real_test']['accuracy']:.4f} "
            f"real_test_macro={metrics_by_name['real_test']['macro_recall']:.4f}"
        )
    return global_step, metrics_by_name


def _run_lbfgs_phase(
    *,
    trainer,
    model: torch.nn.Module,
    train_seed_bank: dict[str, np.ndarray],
    loop_recipe: list[dict[str, float | str]],
    feature_set: str,
    scale,
    eval_sets: dict[str, tuple[torch.Tensor, torch.Tensor]],
    train_points: int,
    batch_size: int,
    learning_rate: float,
    loops: int,
    steps_per_loop: int,
    seed: int,
    device: torch.device,
    history_rows: list[dict[str, float | str]],
    global_step_start: int,
    max_iter: int,
    history_size: int,
) -> tuple[int, dict[str, dict[str, float]]]:
    global_step = global_step_start
    start = time.time()
    for loop_index in range(1, loops + 1):
        strain_batch, branch_batch, material_batch = trainer._draw_training_recipe(
            train_seed_bank,
            recipe=loop_recipe,
            element_count=train_points,
            seed=seed + 20000 * loop_index,
        )
        x_train_np = trainer._build_point_features(strain_batch, material_batch, feature_set=feature_set)
        y_train_np = trainer._flatten_pointwise_labels(branch_batch)
        x_train = torch.from_numpy(scale(x_train_np))
        y_train = torch.from_numpy(y_train_np)

        class_weights = trainer._class_weights(y_train_np)
        binary_weights = trainer._binary_class_weights(y_train_np)
        plastic_weights = trainer._plastic_class_weights(y_train_np)
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=learning_rate,
            max_iter=max_iter,
            history_size=history_size,
            line_search_fn="strong_wolfe",
        )

        perm = torch.randperm(x_train.shape[0])
        for step_index in range(1, steps_per_loop + 1):
            start = ((step_index - 1) * batch_size) % x_train.shape[0]
            stop = start + batch_size
            if stop <= x_train.shape[0]:
                idx = perm[start:stop]
            else:
                idx = torch.cat([perm[start:], perm[: stop - x_train.shape[0]]], dim=0)
            xb = x_train[idx].to(device)
            yb = y_train[idx].to(device)

            def closure() -> torch.Tensor:
                optimizer.zero_grad(set_to_none=True)
                loss = trainer._compute_loss(
                    model,
                    xb,
                    yb,
                    model_type="hierarchical",
                    class_weights=class_weights,
                    binary_weights=binary_weights,
                    plastic_weights=plastic_weights,
                    plastic_loss_weight=1.0,
                )
                loss.backward()
                return loss

            model.train(True)
            loss = optimizer.step(closure)
            metrics_by_name = _evaluate_phase(trainer, model, eval_sets)
            global_step += 1
            row: dict[str, float | str] = {
                "phase": "lbfgs",
                "global_epoch": float(global_step),
                "loop_index": float(loop_index),
                "step_in_loop": float(step_index),
                "train_loss": float(loss.item()),
                "lr": learning_rate,
                "runtime_s": time.time() - start,
            }
            for split_name, split_metrics in metrics_by_name.items():
                for key, value in split_metrics.items():
                    row[f"{split_name}_{key}"] = float(value)
            history_rows.append(row)
            print(
                f"[lbfgs] loop={loop_index}/{loops} step={step_index}/{steps_per_loop} loss={row['train_loss']:.6f} "
                f"syn_val_acc={metrics_by_name['synthetic_val']['accuracy']:.4f} "
                f"syn_val_macro={metrics_by_name['synthetic_val']['macro_recall']:.4f} "
                f"real_test_acc={metrics_by_name['real_test']['accuracy']:.4f} "
                f"real_test_macro={metrics_by_name['real_test']['macro_recall']:.4f}"
            )
    return global_step, metrics_by_name


def _write_report(report_path: Path, *, artifact_dir: Path, summary: dict[str, object]) -> None:
    rel = artifact_dir.relative_to(report_path.parent.parent)
    infl = summary["inflation_check"]
    adam = summary["adam_phase"]
    lbfgs = summary["lbfgs_phase"]
    lines = [
        "# Cover Layer Branch Predictor Inflated Adam Then LBFGS Report",
        "",
        "## Summary",
        "",
        f"- base checkpoint: `{summary['base_checkpoint']}`",
        f"- width inflation: `{summary['old_width']} -> {summary['new_width']}`",
        f"- inflation noise scale: `{summary['inflate_noise_scale']:.1e}`",
        f"- Adam phase: `{adam['loops']}` datasets, lr `{adam['learning_rate']:.1e}`, batch `{adam['batch_size']}`, train points `{adam['train_points']}`",
        f"- LBFGS phase: `{lbfgs['loops']}` datasets, `{lbfgs['steps_per_loop']}` steps each, lr `{lbfgs['learning_rate']:.1e}`",
        "",
        "## Inflation Check",
        "",
        f"- synthetic test accuracy / macro recall: `{infl['base_metrics']['synthetic_test']['accuracy']:.4f}` / `{infl['base_metrics']['synthetic_test']['macro_recall']:.4f}` -> `{infl['inflated_metrics']['synthetic_test']['accuracy']:.4f}` / `{infl['inflated_metrics']['synthetic_test']['macro_recall']:.4f}`",
        f"- real test accuracy / macro recall: `{infl['base_metrics']['real_test']['accuracy']:.4f}` / `{infl['base_metrics']['real_test']['macro_recall']:.4f}` -> `{infl['inflated_metrics']['real_test']['accuracy']:.4f}` / `{infl['inflated_metrics']['real_test']['macro_recall']:.4f}`",
        f"- prediction change rate on synthetic val / synthetic test / real val / real test: "
        f"`{infl['prediction_change_rate']['synthetic_val']:.6f}` / "
        f"`{infl['prediction_change_rate']['synthetic_test']:.6f}` / "
        f"`{infl['prediction_change_rate']['real_val']:.6f}` / "
        f"`{infl['prediction_change_rate']['real_test']:.6f}`",
        "",
        "## Phase Results",
        "",
        f"- after Adam synthetic test accuracy / macro recall: `{adam['metrics']['synthetic_test']['accuracy']:.4f}` / `{adam['metrics']['synthetic_test']['macro_recall']:.4f}`",
        f"- after Adam real test accuracy / macro recall: `{adam['metrics']['real_test']['accuracy']:.4f}` / `{adam['metrics']['real_test']['macro_recall']:.4f}`",
        f"- after LBFGS synthetic test accuracy / macro recall: `{lbfgs['metrics']['synthetic_test']['accuracy']:.4f}` / `{lbfgs['metrics']['synthetic_test']['macro_recall']:.4f}`",
        f"- after LBFGS real test accuracy / macro recall: `{lbfgs['metrics']['real_test']['accuracy']:.4f}` / `{lbfgs['metrics']['real_test']['macro_recall']:.4f}`",
        "",
        "## Checkpoints",
        "",
        f"- inflated checkpoint: `{summary['inflated_checkpoint']}`",
        f"- after Adam: `{adam['checkpoint']}`",
        f"- after LBFGS: `{lbfgs['checkpoint']}`",
        "",
        "## Curves",
        "",
        "![Training history](../" + str(rel / "training_history.png") + ")",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    trainer = _load_trainer_module()

    parser = argparse.ArgumentParser(description="Inflate the cover-layer branch predictor width and continue with Adam then LBFGS.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("experiment_runs/real_sim/cover_layer_branch_predictor_expert_principal_w1024_d6_acceptall_p1_1000_capped2_20260315/best.pt"),
    )
    parser.add_argument(
        "--base-summary-json",
        type=Path,
        default=Path("experiment_runs/real_sim/cover_layer_branch_predictor_expert_principal_w1024_d6_acceptall_p1_1000_capped2_20260315/summary.json"),
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
        default=Path("experiment_runs/real_sim/cover_layer_branch_predictor_expert_principal_w2048_d6_adam_lbfgs_20260316"),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("docs/cover_layer_branch_predictor_expert_principal_w2048_d6_adam_lbfgs.md"),
    )
    parser.add_argument("--summary-json", type=Path, default=Path("experiment_runs/real_sim/cover_layer_branch_predictor_expert_principal_w2048_d6_adam_lbfgs_20260316/summary.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-elements-per-call", type=int, default=128)
    parser.add_argument("--train-seed-calls", type=int, default=24)
    parser.add_argument("--eval-seed-calls", type=int, default=8)
    parser.add_argument("--synthetic-val-elements", type=int, default=16384)
    parser.add_argument("--synthetic-test-elements", type=int, default=16384)
    parser.add_argument("--train-points", type=int, default=81920)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--new-width", type=int, default=2048)
    parser.add_argument("--inflate-noise-scale", type=float, default=1.0e-9)
    parser.add_argument("--adam-loops", type=int, default=10)
    parser.add_argument("--adam-lr", type=float, default=1.0e-6)
    parser.add_argument("--adam-weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--lbfgs-loops", type=int, default=10)
    parser.add_argument("--lbfgs-steps-per-loop", type=int, default=10)
    parser.add_argument("--lbfgs-lr", type=float, default=1.0e-2)
    parser.add_argument("--lbfgs-max-iter", type=int, default=20)
    parser.add_argument("--lbfgs-history-size", type=int, default=50)
    parser.add_argument("--lbfgs-device", choices=("same", "cpu"), default="same")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_summary = json.loads(args.base_summary_json.read_text(encoding="utf-8"))
    base_ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    feature_set = str(base_ckpt["feature_set"])
    model_type = str(base_ckpt.get("model_type", base_summary.get("model_type", "hierarchical")))
    if model_type != "hierarchical":
        raise ValueError(f"This runner currently expects a hierarchical checkpoint, got {model_type!r}.")
    x_mean = np.asarray(base_ckpt["x_mean"], dtype=np.float32)
    x_std = np.asarray(base_ckpt["x_std"], dtype=np.float32)
    x_std = np.where(x_std < 1.0e-6, 1.0, x_std)

    def scale(x: np.ndarray) -> np.ndarray:
        return ((x - x_mean) / x_std).astype(np.float32)

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

    benchmark_recipe = [
        {"fraction": 0.60, "selection": "branch_balanced", "noise_scale": 0.18},
        {"fraction": 0.25, "selection": "boundary_smooth_right", "noise_scale": 0.05},
        {"fraction": 0.15, "selection": "tail", "noise_scale": 0.25},
    ]
    loop_recipe = [
        {"fraction": 0.35, "selection": "branch_balanced", "noise_scale": 0.20},
        {"fraction": 0.30, "selection": "boundary_smooth_right", "noise_scale": 0.05},
        {"fraction": 0.20, "selection": "smooth_edge", "noise_scale": 0.22},
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
    x_syn_val = torch.from_numpy(scale(trainer._build_point_features(strain_syn_val, material_syn_val, feature_set=feature_set))).to(device)
    y_syn_val = torch.from_numpy(trainer._flatten_pointwise_labels(branch_syn_val)).to(device)
    x_syn_test = torch.from_numpy(scale(trainer._build_point_features(strain_syn_test, material_syn_test, feature_set=feature_set))).to(device)
    y_syn_test = torch.from_numpy(trainer._flatten_pointwise_labels(branch_syn_test)).to(device)

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
    x_real_val = torch.from_numpy(scale(trainer._build_point_features(strain_real_val, material_real_val, feature_set=feature_set))).to(device)
    y_real_val = torch.from_numpy(trainer._flatten_pointwise_labels(branch_real_val)).to(device)
    x_real_test = torch.from_numpy(scale(trainer._build_point_features(strain_real_test, material_real_test, feature_set=feature_set))).to(device)
    y_real_test = torch.from_numpy(trainer._flatten_pointwise_labels(branch_real_test)).to(device)

    eval_sets = {
        "synthetic_val": (x_syn_val, y_syn_val),
        "synthetic_test": (x_syn_test, y_syn_test),
        "real_val": (x_real_val, y_real_val),
        "real_test": (x_real_test, y_real_test),
    }

    old_model = trainer.HierarchicalBranchNet(
        in_dim=int(base_ckpt["input_dim"]),
        width=int(base_ckpt["width"]),
        depth=int(base_ckpt["depth"]),
    ).to(device)
    old_model.load_state_dict(base_ckpt["state_dict"])
    base_metrics = _evaluate_phase(trainer, old_model, eval_sets)

    inflated_ckpt = _inflate_checkpoint(
        base_ckpt,
        trainer,
        new_width=args.new_width,
        noise_scale=args.inflate_noise_scale,
        seed=args.seed + 123,
    )
    model = trainer.HierarchicalBranchNet(
        in_dim=int(inflated_ckpt["input_dim"]),
        width=int(inflated_ckpt["width"]),
        depth=int(inflated_ckpt["depth"]),
    ).to(device)
    model.load_state_dict(inflated_ckpt["state_dict"])
    inflated_metrics = _evaluate_phase(trainer, model, eval_sets)
    prediction_change_rate = {
        "synthetic_val": _prediction_change_rate(trainer, old_model, model, x_syn_val),
        "synthetic_test": _prediction_change_rate(trainer, old_model, model, x_syn_test),
        "real_val": _prediction_change_rate(trainer, old_model, model, x_real_val),
        "real_test": _prediction_change_rate(trainer, old_model, model, x_real_test),
    }

    inflated_ckpt_path = args.output_dir / "inflated_init.pt"
    _save_checkpoint(
        inflated_ckpt_path,
        base_ckpt=inflated_ckpt,
        model=model,
        width=args.new_width,
        phase_name="inflated_init",
    )

    history_rows: list[dict[str, float | str]] = []
    global_step = 0
    global_step, adam_metrics = _run_adam_phase(
        trainer=trainer,
        model=model,
        train_seed_bank=train_seed_bank,
        loop_recipe=loop_recipe,
        feature_set=feature_set,
        scale=scale,
        eval_sets=eval_sets,
        train_points=args.train_points,
        batch_size=args.batch_size,
        learning_rate=args.adam_lr,
        weight_decay=args.adam_weight_decay,
        loops=args.adam_loops,
        seed=args.seed + 1000,
        device=device,
        history_rows=history_rows,
        global_step_start=global_step,
    )
    adam_ckpt_path = args.output_dir / "after_adam.pt"
    _save_checkpoint(adam_ckpt_path, base_ckpt=inflated_ckpt, model=model, width=args.new_width, phase_name="after_adam")

    lbfgs_device = device if args.lbfgs_device == "same" else torch.device("cpu")
    if lbfgs_device != device:
        model = model.to(lbfgs_device)
        eval_sets = _move_eval_sets(eval_sets, lbfgs_device)

    global_step, lbfgs_metrics = _run_lbfgs_phase(
        trainer=trainer,
        model=model,
        train_seed_bank=train_seed_bank,
        loop_recipe=loop_recipe,
        feature_set=feature_set,
        scale=scale,
        eval_sets=eval_sets,
        train_points=args.train_points,
        batch_size=args.batch_size,
        learning_rate=args.lbfgs_lr,
        loops=args.lbfgs_loops,
        steps_per_loop=args.lbfgs_steps_per_loop,
        seed=args.seed + 2000,
        device=lbfgs_device,
        history_rows=history_rows,
        global_step_start=global_step,
        max_iter=args.lbfgs_max_iter,
        history_size=args.lbfgs_history_size,
    )
    lbfgs_ckpt_path = args.output_dir / "after_lbfgs.pt"
    _save_checkpoint(lbfgs_ckpt_path, base_ckpt=inflated_ckpt, model=model, width=args.new_width, phase_name="after_lbfgs")

    trainer._plot_history(history_rows, args.output_dir / "training_history.png")
    (args.output_dir / "history.json").write_text(json.dumps(history_rows, indent=2), encoding="utf-8")

    summary = {
        "base_checkpoint": str(args.checkpoint),
        "old_width": int(base_ckpt["width"]),
        "new_width": args.new_width,
        "inflate_noise_scale": args.inflate_noise_scale,
        "inflated_checkpoint": str(inflated_ckpt_path),
        "inflation_check": {
            "base_metrics": base_metrics,
            "inflated_metrics": inflated_metrics,
            "prediction_change_rate": prediction_change_rate,
        },
        "adam_phase": {
            "loops": args.adam_loops,
            "learning_rate": args.adam_lr,
            "batch_size": args.batch_size,
            "train_points": args.train_points,
            "metrics": adam_metrics,
            "checkpoint": str(adam_ckpt_path),
        },
        "lbfgs_phase": {
            "loops": args.lbfgs_loops,
            "steps_per_loop": args.lbfgs_steps_per_loop,
            "learning_rate": args.lbfgs_lr,
            "train_points": args.train_points,
            "metrics": lbfgs_metrics,
            "checkpoint": str(lbfgs_ckpt_path),
        },
    }
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_report(args.report_path, artifact_dir=args.output_dir, summary=summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
