"""Inference helpers and constitutive-style surrogate wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .branch_geometry import (
    branch_min_term_names,
    compute_branch_geometry_principal,
    global_min_term_names,
    select_branch_conditioned_distance,
)
from .materials import build_reduced_material_from_raw
from .models import compute_trial_stress, spectral_decomposition_from_strain
from .mohr_coulomb import (
    BRANCH_NAMES,
    constitutive_update_3d,
    yield_violation_rel_principal_3d,
)
from .training import predict_with_checkpoint
from .voigt import stress_voigt_to_tensor


def _coerce_strain_and_material(
    strain_eng: np.ndarray,
    material_reduced: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    strain = np.asarray(strain_eng, dtype=float)
    if strain.ndim == 1:
        strain = strain[None, :]
    material = np.asarray(material_reduced, dtype=float)
    if material.ndim != 2 or material.shape[0] != strain.shape[0]:
        raise ValueError("material_reduced must have shape (n, 5) matching strain_eng.")
    return strain, material


def normalized_branch_entropy(
    branch_probabilities: np.ndarray,
    *,
    plastic_only: bool = True,
    eps: float = 1.0e-12,
) -> np.ndarray:
    """Return entropy normalized to `[0, 1]`."""
    probs = np.asarray(branch_probabilities, dtype=float)
    if probs.ndim != 2:
        raise ValueError(f"Expected branch_probabilities shape (n, k), got {probs.shape}.")
    if probs.shape[1] == 0:
        return np.ones(probs.shape[0], dtype=np.float32)
    if plastic_only and probs.shape[1] >= 5:
        work = probs[:, 1:]
    else:
        work = probs
    total = np.sum(work, axis=1, keepdims=True)
    safe = np.divide(work, np.maximum(total, eps), out=np.zeros_like(work), where=total > eps)
    logk = np.log(max(safe.shape[1], 2))
    entropy = -np.sum(np.where(safe > 0.0, safe * np.log(np.maximum(safe, eps)), 0.0), axis=1)
    normalized = entropy / logk
    normalized = np.where(total.reshape(-1) > eps, normalized, 1.0)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def _principal_from_stress(stress_voigt: np.ndarray) -> np.ndarray:
    return np.linalg.eigvalsh(stress_voigt_to_tensor(stress_voigt))[:, ::-1].astype(np.float32)


def prepare_hybrid_gate_inputs(
    checkpoint_path: str | Path,
    strain_eng: np.ndarray,
    material_reduced: np.ndarray,
    *,
    device: str = "cpu",
    batch_size: int | None = None,
    include_exact: bool = True,
) -> dict[str, Any]:
    """
    Precompute exact geometry, raw model predictions, and gate diagnostics.

    The returned dictionary is intended to be reused across multiple gate
    thresholds so experiment scripts do not need to reload the checkpoint or
    rerun the network for every threshold.
    """
    strain, material = _coerce_strain_and_material(strain_eng, material_reduced)
    trial_stress = compute_trial_stress(strain, material)
    strain_principal, eigvecs = spectral_decomposition_from_strain(strain)
    trial_principal = np.linalg.eigvalsh(stress_voigt_to_tensor(trial_stress))[:, ::-1].astype(np.float32)

    geom = compute_branch_geometry_principal(
        strain_principal,
        c_bar=material[:, 0],
        sin_phi=material[:, 1],
        shear=material[:, 2],
        bulk=material[:, 3],
        lame=material[:, 4],
    )
    d_geom_term = global_min_term_names(geom)
    d_smooth_term = branch_min_term_names(geom, "smooth")
    d_left_term = branch_min_term_names(geom, "left_edge")
    d_right_term = branch_min_term_names(geom, "right_edge")
    d_apex_term = branch_min_term_names(geom, "apex")

    raw_pred = predict_with_checkpoint(
        checkpoint_path,
        strain,
        material,
        device=device,
        batch_size=batch_size,
    )
    raw_stress = raw_pred["stress"].astype(np.float32)
    raw_principal = raw_pred.get("stress_principal")
    if raw_principal is None:
        raw_principal = _principal_from_stress(raw_stress)
    else:
        raw_principal = raw_principal.astype(np.float32)

    branch_probabilities = raw_pred.get("branch_probabilities")
    if branch_probabilities is None:
        branch_probabilities = np.zeros((strain.shape[0], len(BRANCH_NAMES)), dtype=np.float32)
    else:
        branch_probabilities = branch_probabilities.astype(np.float32)

    plastic_branch_probabilities = branch_probabilities[:, 1:] if branch_probabilities.shape[1] >= 5 else branch_probabilities
    branch_entropy = normalized_branch_entropy(branch_probabilities, plastic_only=True)
    predicted_branch_id = np.argmax(plastic_branch_probabilities, axis=1).astype(np.int8) + 1
    predicted_branch_name = np.asarray([BRANCH_NAMES[idx] for idx in predicted_branch_id], dtype=object)

    chosen_distance_pred, chosen_family_pred, chosen_term_pred = select_branch_conditioned_distance(geom, predicted_branch_id)
    raw_yield_violation = yield_violation_rel_principal_3d(
        raw_principal,
        c_bar=material[:, 0],
        sin_phi=material[:, 1],
    ).astype(np.float32)

    correction_norm = np.linalg.norm(raw_stress - trial_stress, axis=1)
    correction_denom = np.linalg.norm(raw_stress, axis=1) + np.maximum(material[:, 0], 0.0) + 1.0e-12
    predicted_correction_norm_rel = (correction_norm / correction_denom).astype(np.float32)

    prepared: dict[str, Any] = {
        "checkpoint_path": str(checkpoint_path),
        "strain_eng": strain.astype(np.float32),
        "material_reduced": material.astype(np.float32),
        "strain_principal": strain_principal.astype(np.float32),
        "eigvecs": eigvecs.astype(np.float32),
        "trial_stress": trial_stress.astype(np.float32),
        "trial_principal": trial_principal.astype(np.float32),
        "raw_stress": raw_stress.astype(np.float32),
        "raw_stress_principal": raw_principal.astype(np.float32),
        "branch_probabilities": branch_probabilities.astype(np.float32),
        "branch_entropy": branch_entropy.astype(np.float32),
        "predicted_branch_id": predicted_branch_id.astype(np.int8),
        "predicted_branch_name": predicted_branch_name,
        "predicted_branch_distance": chosen_distance_pred.astype(np.float32),
        "predicted_branch_family": chosen_family_pred,
        "predicted_branch_min_term": chosen_term_pred,
        "raw_yield_violation_rel": raw_yield_violation.astype(np.float32),
        "predicted_correction_norm_rel": predicted_correction_norm_rel.astype(np.float32),
        "f_trial": geom.f_trial.astype(np.float32),
        "m_yield": geom.m_yield.astype(np.float32),
        "m_smooth_left": geom.m_smooth_left.astype(np.float32),
        "m_smooth_right": geom.m_smooth_right.astype(np.float32),
        "m_left_apex": geom.m_left_apex.astype(np.float32),
        "m_right_apex": geom.m_right_apex.astype(np.float32),
        "gap12_norm": geom.gap12_norm.astype(np.float32),
        "gap23_norm": geom.gap23_norm.astype(np.float32),
        "gap_min_norm": geom.gap_min_norm.astype(np.float32),
        "d_geom": geom.d_geom.astype(np.float32),
        "d_smooth": geom.d_smooth.astype(np.float32),
        "d_left": geom.d_left.astype(np.float32),
        "d_right": geom.d_right.astype(np.float32),
        "d_apex": geom.d_apex.astype(np.float32),
        "d_geom_min_term": d_geom_term,
        "d_smooth_min_term": d_smooth_term,
        "d_left_min_term": d_left_term,
        "d_right_min_term": d_right_term,
        "d_apex_min_term": d_apex_term,
        "elastic_mask": (geom.f_trial <= 0.0),
    }

    if include_exact:
        exact = constitutive_update_3d(
            strain,
            c_bar=material[:, 0],
            sin_phi=material[:, 1],
            shear=material[:, 2],
            bulk=material[:, 3],
            lame=material[:, 4],
        )
        prepared["exact_stress"] = exact.stress.astype(np.float32)
        prepared["exact_stress_principal"] = exact.stress_principal.astype(np.float32)
        prepared["true_branch_id"] = exact.branch_id.astype(np.int8)
    return prepared


def apply_hybrid_gate(
    prepared: dict[str, Any],
    *,
    delta_geom: float,
    gate_mode: str = "global",
    entropy_threshold: float | None = None,
    branch_id_for_gate: np.ndarray | None = None,
    rejector_score: np.ndarray | None = None,
    rejector_threshold: float | None = None,
) -> dict[str, Any]:
    """
    Apply one routing policy to precomputed hybrid gate inputs.

    Supported `gate_mode` values:
    - `global`
    - `oracle_branch`
    - `predicted_branch`
    - `rejector`
    """
    n = prepared["strain_eng"].shape[0]
    stress = np.zeros((n, 6), dtype=np.float32)
    stress_principal = np.zeros((n, 3), dtype=np.float32)
    elastic_mask = prepared["elastic_mask"].astype(bool)
    fallback_mask = np.zeros(n, dtype=bool)
    learned_mask = np.zeros(n, dtype=bool)
    route_reason = np.full(n, "elastic_exact", dtype=object)
    chosen_gate_family = np.full(n, "elastic", dtype=object)
    chosen_branch_distance = np.full(n, np.nan, dtype=np.float32)
    min_term_name = np.full(n, "elastic", dtype=object)
    accepted_yield_violation = np.full(n, np.nan, dtype=np.float32)

    stress[elastic_mask] = prepared["trial_stress"][elastic_mask]
    stress_principal[elastic_mask] = prepared["trial_principal"][elastic_mask]

    plastic_mask = ~elastic_mask
    if np.any(plastic_mask):
        if gate_mode == "global":
            distance = prepared["d_geom"].astype(np.float32)
            family = np.full(n, "global", dtype=object)
            term = prepared["d_geom_min_term"].astype(object)
            entropy_fallback = np.zeros(n, dtype=bool)
        elif gate_mode == "oracle_branch":
            gate_branch = branch_id_for_gate
            if gate_branch is None:
                gate_branch = prepared.get("true_branch_id")
            if gate_branch is None:
                raise ValueError("oracle_branch routing requires branch_id_for_gate or prepared['true_branch_id'].")
            gate_branch = np.asarray(gate_branch, dtype=np.int64).reshape(-1)
            if gate_branch.size == 1 and n > 1:
                gate_branch = np.full(n, int(gate_branch.item()), dtype=np.int64)
            if gate_branch.shape[0] != n:
                raise ValueError(f"branch_id_for_gate shape {gate_branch.shape} does not match batch {n}.")
            distance = np.full(n, np.nan, dtype=np.float32)
            family = np.full(n, "", dtype=object)
            term = np.full(n, "", dtype=object)
            oracle_sources = {
                1: ("smooth", prepared["d_smooth"], prepared["d_smooth_min_term"]),
                2: ("left_edge", prepared["d_left"], prepared["d_left_min_term"]),
                3: ("right_edge", prepared["d_right"], prepared["d_right_min_term"]),
                4: ("apex", prepared["d_apex"], prepared["d_apex_min_term"]),
            }
            for branch_id, (family_name, branch_distance, branch_term) in oracle_sources.items():
                mask = gate_branch == branch_id
                if np.any(mask):
                    distance[mask] = branch_distance[mask]
                    family[mask] = family_name
                    term[mask] = branch_term[mask]
            entropy_fallback = np.zeros(n, dtype=bool)
        elif gate_mode == "predicted_branch":
            if entropy_threshold is None:
                raise ValueError("predicted_branch routing requires entropy_threshold.")
            distance = prepared["predicted_branch_distance"].astype(np.float32)
            family = prepared["predicted_branch_family"].astype(object)
            term = prepared["predicted_branch_min_term"].astype(object)
            entropy_fallback = prepared["branch_entropy"] > float(entropy_threshold)
        elif gate_mode == "rejector":
            if rejector_score is None or rejector_threshold is None:
                raise ValueError("rejector routing requires rejector_score and rejector_threshold.")
            score = np.asarray(rejector_score, dtype=float).reshape(-1)
            if score.shape[0] != n:
                raise ValueError(f"rejector_score shape {score.shape} does not match batch {n}.")
            distance = prepared["predicted_branch_distance"].astype(np.float32)
            family = np.full(n, "rejector", dtype=object)
            term = prepared["predicted_branch_min_term"].astype(object)
            entropy_fallback = score >= float(rejector_threshold)
        else:
            raise ValueError(f"Unsupported gate_mode {gate_mode!r}.")

        chosen_gate_family[plastic_mask] = family[plastic_mask]
        chosen_branch_distance[plastic_mask] = distance[plastic_mask]
        min_term_name[plastic_mask] = term[plastic_mask]

        distance_fallback = np.zeros(n, dtype=bool)
        if gate_mode != "rejector":
            distance_fallback[plastic_mask] = distance[plastic_mask] < float(delta_geom)
        fallback_mask = plastic_mask & (distance_fallback | entropy_fallback)
        learned_mask = plastic_mask & ~fallback_mask

        route_reason[learned_mask] = "learned"
        route_reason[plastic_mask & distance_fallback] = "distance_fallback"
        if gate_mode == "rejector":
            route_reason[plastic_mask & entropy_fallback] = "rejector_fallback"
        else:
            route_reason[plastic_mask & entropy_fallback] = "entropy_fallback"

        if "exact_stress" in prepared:
            stress[fallback_mask] = prepared["exact_stress"][fallback_mask]
            stress_principal[fallback_mask] = prepared["exact_stress_principal"][fallback_mask]
        else:
            exact = constitutive_update_3d(
                prepared["strain_eng"][fallback_mask],
                c_bar=prepared["material_reduced"][fallback_mask, 0],
                sin_phi=prepared["material_reduced"][fallback_mask, 1],
                shear=prepared["material_reduced"][fallback_mask, 2],
                bulk=prepared["material_reduced"][fallback_mask, 3],
                lame=prepared["material_reduced"][fallback_mask, 4],
            )
            stress[fallback_mask] = exact.stress.astype(np.float32)
            stress_principal[fallback_mask] = exact.stress_principal.astype(np.float32)

        stress[learned_mask] = prepared["raw_stress"][learned_mask]
        stress_principal[learned_mask] = prepared["raw_stress_principal"][learned_mask]
        accepted_yield_violation[learned_mask] = prepared["raw_yield_violation_rel"][learned_mask]

    route_counts = {
        "elastic": int(np.sum(elastic_mask)),
        "fallback": int(np.sum(fallback_mask)),
        "learned": int(np.sum(learned_mask)),
    }
    predicted_branch_counts = {
        BRANCH_NAMES[idx]: int(np.sum(prepared["predicted_branch_id"] == idx))
        for idx in range(1, len(BRANCH_NAMES))
    }
    learned_predicted_branch_counts = {
        BRANCH_NAMES[idx]: int(np.sum(learned_mask & (prepared["predicted_branch_id"] == idx)))
        for idx in range(1, len(BRANCH_NAMES))
    }
    return {
        "stress": stress.astype(np.float32),
        "stress_principal": stress_principal.astype(np.float32),
        "elastic_mask": elastic_mask.astype(bool),
        "fallback_mask": fallback_mask.astype(bool),
        "learned_mask": learned_mask.astype(bool),
        "route_reason": route_reason,
        "chosen_gate_family": chosen_gate_family,
        "chosen_branch_distance": chosen_branch_distance.astype(np.float32),
        "min_term_name": min_term_name,
        "branch_probabilities": prepared["branch_probabilities"].astype(np.float32),
        "branch_entropy": prepared["branch_entropy"].astype(np.float32),
        "predicted_branch_id": prepared["predicted_branch_id"].astype(np.int8),
        "predicted_branch_name": prepared["predicted_branch_name"],
        "accepted_yield_violation_rel": accepted_yield_violation.astype(np.float32),
        "d_geom": prepared["d_geom"].astype(np.float32),
        "d_smooth": prepared["d_smooth"].astype(np.float32),
        "d_left": prepared["d_left"].astype(np.float32),
        "d_right": prepared["d_right"].astype(np.float32),
        "d_apex": prepared["d_apex"].astype(np.float32),
        "route_counts": route_counts,
        "predicted_branch_counts": predicted_branch_counts,
        "learned_predicted_branch_counts": learned_predicted_branch_counts,
    }


def hybrid_predict_with_checkpoint(
    checkpoint_path: str | Path,
    strain_eng: np.ndarray,
    material_reduced: np.ndarray,
    *,
    delta_geom: float,
    device: str = "cpu",
    batch_size: int | None = None,
    gate_mode: str = "global",
    entropy_threshold: float | None = None,
    branch_id_for_gate: np.ndarray | None = None,
    rejector_score: np.ndarray | None = None,
    rejector_threshold: float | None = None,
) -> dict[str, Any]:
    """Convenience wrapper: prepare inputs and apply one hybrid routing policy."""
    prepared = prepare_hybrid_gate_inputs(
        checkpoint_path,
        strain_eng,
        material_reduced,
        device=device,
        batch_size=batch_size,
        include_exact=True,
    )
    return apply_hybrid_gate(
        prepared,
        delta_geom=delta_geom,
        gate_mode=gate_mode,
        entropy_threshold=entropy_threshold,
        branch_id_for_gate=branch_id_for_gate,
        rejector_score=rejector_score,
        rejector_threshold=rejector_threshold,
    )


@dataclass
class ConstitutiveSurrogate:
    """Load a checkpoint and expose a constitutive-operator-like prediction API."""

    checkpoint_path: str
    device: str = "cpu"

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path, device: str = "cpu") -> "ConstitutiveSurrogate":
        return cls(checkpoint_path=str(checkpoint_path), device=device)

    def predict_reduced(
        self,
        strain_eng: np.ndarray,
        *,
        c_bar: np.ndarray | float,
        sin_phi: np.ndarray | float,
        shear: np.ndarray | float,
        bulk: np.ndarray | float,
        lame: np.ndarray | float,
    ) -> dict[str, np.ndarray]:
        strain = np.asarray(strain_eng, dtype=float)
        if strain.ndim == 1:
            strain = strain[None, :]
        n = strain.shape[0]
        material_reduced = np.column_stack(
            [
                np.broadcast_to(np.asarray(c_bar, dtype=float), (n,)).reshape(-1),
                np.broadcast_to(np.asarray(sin_phi, dtype=float), (n,)).reshape(-1),
                np.broadcast_to(np.asarray(shear, dtype=float), (n,)).reshape(-1),
                np.broadcast_to(np.asarray(bulk, dtype=float), (n,)).reshape(-1),
                np.broadcast_to(np.asarray(lame, dtype=float), (n,)).reshape(-1),
            ]
        )
        return predict_with_checkpoint(
            self.checkpoint_path,
            strain,
            material_reduced,
            device=self.device,
        )

    def predict_raw(
        self,
        strain_eng: np.ndarray,
        *,
        c0: np.ndarray | float,
        phi_rad: np.ndarray | float,
        psi_rad: np.ndarray | float,
        young: np.ndarray | float,
        poisson: np.ndarray | float,
        strength_reduction: np.ndarray | float,
        davis_type: str | int | list[str | int],
    ) -> dict[str, np.ndarray]:
        reduced = build_reduced_material_from_raw(
            c0=c0,
            phi_rad=phi_rad,
            psi_rad=psi_rad,
            young=young,
            poisson=poisson,
            strength_reduction=strength_reduction,
            davis_type=davis_type,
        )
        return self.predict_reduced(
            strain_eng,
            c_bar=reduced.c_bar,
            sin_phi=reduced.sin_phi,
            shear=reduced.shear,
            bulk=reduced.bulk,
            lame=reduced.lame,
        )


@dataclass
class HybridConstitutiveSurrogate(ConstitutiveSurrogate):
    """Checkpoint-backed constitutive wrapper with exact elastic/fallback logic."""

    delta_geom: float = 0.02
    gate_mode: str = "global"
    entropy_threshold: float | None = None

    def predict_hybrid_reduced(
        self,
        strain_eng: np.ndarray,
        *,
        c_bar: np.ndarray | float,
        sin_phi: np.ndarray | float,
        shear: np.ndarray | float,
        bulk: np.ndarray | float,
        lame: np.ndarray | float,
        batch_size: int | None = None,
        branch_id_for_gate: np.ndarray | None = None,
        rejector_score: np.ndarray | None = None,
        rejector_threshold: float | None = None,
    ) -> dict[str, Any]:
        strain = np.asarray(strain_eng, dtype=float)
        if strain.ndim == 1:
            strain = strain[None, :]
        n = strain.shape[0]
        material_reduced = np.column_stack(
            [
                np.broadcast_to(np.asarray(c_bar, dtype=float), (n,)).reshape(-1),
                np.broadcast_to(np.asarray(sin_phi, dtype=float), (n,)).reshape(-1),
                np.broadcast_to(np.asarray(shear, dtype=float), (n,)).reshape(-1),
                np.broadcast_to(np.asarray(bulk, dtype=float), (n,)).reshape(-1),
                np.broadcast_to(np.asarray(lame, dtype=float), (n,)).reshape(-1),
            ]
        )
        return hybrid_predict_with_checkpoint(
            self.checkpoint_path,
            strain,
            material_reduced,
            delta_geom=self.delta_geom,
            device=self.device,
            batch_size=batch_size,
            gate_mode=self.gate_mode,
            entropy_threshold=self.entropy_threshold,
            branch_id_for_gate=branch_id_for_gate,
            rejector_score=rejector_score,
            rejector_threshold=rejector_threshold,
        )
