"""Exact and softmin projections in ordered principal-stress space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

PROJECTION_CANDIDATE_NAMES = ("pass_through", "smooth", "left_edge", "right_edge", "apex")
PROJECTION_CANDIDATE_TO_ID = {name: i for i, name in enumerate(PROJECTION_CANDIDATE_NAMES)}
__all__ = [
    "PROJECTION_CANDIDATE_NAMES",
    "PROJECTION_CANDIDATE_TO_ID",
    "PrincipalProjectionDetails",
    "mc_projection_candidate_details",
    "principal_mc_projection_candidates",
    "principal_mc_projection_candidates_torch",
    "project_mc_principal_numpy",
    "project_mc_principal_torch",
    "project_principal_mc",
    "project_principal_mc_torch",
]


@dataclass
class PrincipalProjectionDetails:
    """Candidate-level details for the principal-space projector."""

    principal_input: Any
    principal_sorted: Any
    c_bar: Any
    sin_phi: Any
    projected_stress: Any
    candidate_names: tuple[str, ...]
    candidate_principal: Any
    feasible_mask: Any
    candidate_sq_distance: Any
    selected_index: Any
    selected_label: Any
    displacement_norm: Any
    soft_weights: Any | None = None


def _coerce_principal_numpy(principal: np.ndarray) -> np.ndarray:
    arr = np.asarray(principal, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Expected principal shape (n, 3), got {arr.shape}.")
    return arr


def _coerce_principal_torch(principal: torch.Tensor) -> torch.Tensor:
    arr = torch.as_tensor(principal)
    if not torch.is_floating_point(arr):
        arr = arr.to(dtype=torch.get_default_dtype())
    if arr.ndim == 1:
        arr = arr.unsqueeze(0)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Expected principal shape (n, 3), got {tuple(arr.shape)}.")
    return arr


def _broadcast_numpy(n: int, c_bar: np.ndarray | float, sin_phi: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    c, s = np.broadcast_arrays(np.asarray(c_bar, dtype=float), np.asarray(sin_phi, dtype=float))
    c = np.asarray(c, dtype=float).reshape(-1)
    s = np.asarray(s, dtype=float).reshape(-1)
    if c.size == 1:
        c = np.full(n, c.item(), dtype=float)
    if s.size == 1:
        s = np.full(n, s.item(), dtype=float)
    if c.shape[0] != n or s.shape[0] != n:
        raise ValueError(f"Material batch mismatch: got {c.shape[0]} and {s.shape[0]}, expected {n}.")
    return c, s


def _broadcast_torch(
    n: int,
    c_bar: torch.Tensor | float,
    sin_phi: torch.Tensor | float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    c = torch.as_tensor(c_bar, dtype=dtype, device=device)
    s = torch.as_tensor(sin_phi, dtype=dtype, device=device)
    c, s = torch.broadcast_tensors(c, s)
    c = c.reshape(-1)
    s = s.reshape(-1)
    if c.numel() == 1:
        c = c.expand(n).clone()
    if s.numel() == 1:
        s = s.expand(n).clone()
    if c.shape[0] != n or s.shape[0] != n:
        raise ValueError(f"Material batch mismatch: got {c.shape[0]} and {s.shape[0]}, expected {n}.")
    return c, s


def _sort_desc_numpy(principal: np.ndarray) -> np.ndarray:
    return np.sort(principal, axis=1)[:, ::-1]


def _sort_desc_torch(principal: torch.Tensor) -> torch.Tensor:
    return torch.sort(principal, dim=1, descending=True).values


def _yield_function_numpy(principal: np.ndarray, c_bar: np.ndarray, sin_phi: np.ndarray) -> np.ndarray:
    return (1.0 + sin_phi) * principal[:, 0] - (1.0 - sin_phi) * principal[:, 2] - c_bar


def _yield_function_torch(principal: torch.Tensor, c_bar: torch.Tensor, sin_phi: torch.Tensor) -> torch.Tensor:
    return (1.0 + sin_phi) * principal[:, 0] - (1.0 - sin_phi) * principal[:, 2] - c_bar


def _project_affine_numpy(principal: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    residual = (a @ principal[..., None]).squeeze(-1) - b
    gram = a @ np.swapaxes(a, 1, 2)
    lam = np.linalg.solve(gram, residual[..., None]).squeeze(-1)
    return principal - (np.swapaxes(a, 1, 2) @ lam[..., None]).squeeze(-1)


def _project_affine_torch(principal: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    residual = (a @ principal.unsqueeze(-1)).squeeze(-1) - b
    gram = a @ a.transpose(-1, -2)
    lam = torch.linalg.solve(gram, residual.unsqueeze(-1)).squeeze(-1)
    return principal - (a.transpose(-1, -2) @ lam.unsqueeze(-1)).squeeze(-1)


def _candidate_arrays_numpy(
    principal: np.ndarray,
    c_bar: np.ndarray,
    sin_phi: np.ndarray,
    *,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = principal.shape[0]
    s = sin_phi
    c = c_bar
    x = principal

    candidate = np.empty((n, 5, 3), dtype=float)
    candidate[:, 0, :] = x

    denom_s = 2.0 * (1.0 + s * s)
    f_trial = _yield_function_numpy(x, c, s)
    delta = f_trial / denom_s
    candidate[:, 1, 0] = x[:, 0] - delta * (1.0 + s)
    candidate[:, 1, 1] = x[:, 1]
    candidate[:, 1, 2] = x[:, 2] + delta * (1.0 - s)

    a_left = np.zeros((n, 2, 3), dtype=float)
    a_left[:, 0, 0] = 1.0
    a_left[:, 0, 1] = -1.0
    a_left[:, 1, 0] = 1.0 + s
    a_left[:, 1, 2] = -(1.0 - s)
    b_left = np.column_stack([np.zeros(n, dtype=float), c])
    candidate[:, 2, :] = _project_affine_numpy(x, a_left, b_left)

    a_right = np.zeros((n, 2, 3), dtype=float)
    a_right[:, 0, 1] = 1.0
    a_right[:, 0, 2] = -1.0
    a_right[:, 1, 0] = 1.0 + s
    a_right[:, 1, 2] = -(1.0 - s)
    b_right = np.column_stack([np.zeros(n, dtype=float), c])
    candidate[:, 3, :] = _project_affine_numpy(x, a_right, b_right)

    apex_valid = np.abs(s) > tol
    candidate[:, 4, :] = np.nan
    if np.any(apex_valid):
        candidate[apex_valid, 4, :] = c[apex_valid, None] / (2.0 * s[apex_valid, None])

    gap12 = candidate[:, :, 0] - candidate[:, :, 1]
    gap23 = candidate[:, :, 1] - candidate[:, :, 2]
    yield_value = _yield_function_numpy(candidate.reshape(-1, 3), np.repeat(c, 5), np.repeat(s, 5)).reshape(n, 5)
    sq_distance = np.sum((candidate - x[:, None, :]) ** 2, axis=-1)
    sq_distance[:, 4] = np.where(apex_valid, sq_distance[:, 4], np.inf)

    admissible = yield_value <= tol
    admissible &= gap12 >= -tol
    admissible &= gap23 >= -tol
    admissible[:, 4] &= apex_valid
    sq_distance = np.where(admissible, sq_distance, np.inf)
    return candidate, sq_distance, yield_value, gap12, gap23, admissible


def _candidate_arrays_torch(
    principal: torch.Tensor,
    c_bar: torch.Tensor,
    sin_phi: torch.Tensor,
    *,
    tol: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = principal.shape[0]
    s = sin_phi
    c = c_bar
    x = principal

    candidate = torch.empty((n, 5, 3), dtype=x.dtype, device=x.device)
    candidate[:, 0, :] = x

    denom_s = 2.0 * (1.0 + s * s)
    f_trial = _yield_function_torch(x, c, s)
    delta = f_trial / denom_s
    candidate[:, 1, 0] = x[:, 0] - delta * (1.0 + s)
    candidate[:, 1, 1] = x[:, 1]
    candidate[:, 1, 2] = x[:, 2] + delta * (1.0 - s)

    a_left = torch.zeros((n, 2, 3), dtype=x.dtype, device=x.device)
    a_left[:, 0, 0] = 1.0
    a_left[:, 0, 1] = -1.0
    a_left[:, 1, 0] = 1.0 + s
    a_left[:, 1, 2] = -(1.0 - s)
    b_left = torch.stack([torch.zeros_like(c), c], dim=1)
    candidate[:, 2, :] = _project_affine_torch(x, a_left, b_left)

    a_right = torch.zeros((n, 2, 3), dtype=x.dtype, device=x.device)
    a_right[:, 0, 1] = 1.0
    a_right[:, 0, 2] = -1.0
    a_right[:, 1, 0] = 1.0 + s
    a_right[:, 1, 2] = -(1.0 - s)
    b_right = torch.stack([torch.zeros_like(c), c], dim=1)
    candidate[:, 3, :] = _project_affine_torch(x, a_right, b_right)

    apex_valid = torch.abs(s) > tol
    candidate[:, 4, :] = torch.nan
    if torch.any(apex_valid):
        candidate[apex_valid, 4, :] = c[apex_valid, None] / (2.0 * s[apex_valid, None])

    gap12 = candidate[:, :, 0] - candidate[:, :, 1]
    gap23 = candidate[:, :, 1] - candidate[:, :, 2]
    yield_value = _yield_function_torch(candidate.reshape(-1, 3), c.repeat_interleave(5), s.repeat_interleave(5)).reshape(n, 5)
    sq_distance = torch.sum((candidate - x[:, None, :]) ** 2, dim=-1)
    sq_distance[:, 4] = torch.where(apex_valid, sq_distance[:, 4], torch.full_like(sq_distance[:, 4], float("inf")))

    admissible = yield_value <= tol
    admissible &= gap12 >= -tol
    admissible &= gap23 >= -tol
    admissible[:, 4] &= apex_valid
    sq_distance = torch.where(admissible, sq_distance, torch.full_like(sq_distance, float("inf")))
    return candidate, sq_distance, yield_value, gap12, gap23, admissible


def _select_exact_numpy(candidate_sq_distance: np.ndarray, candidate_admissible: np.ndarray, *, tol: float) -> np.ndarray:
    n, k = candidate_sq_distance.shape
    selected = np.zeros(n, dtype=np.int64)
    best_dist = np.full(n, np.inf, dtype=float)
    have_best = np.zeros(n, dtype=bool)
    tie_tol = max(float(tol), 1.0e-15)
    for i in range(k):
        dist = candidate_sq_distance[:, i]
        valid = candidate_admissible[:, i]
        better = valid & (~have_best | (dist < best_dist - tie_tol))
        selected = np.where(better, i, selected)
        best_dist = np.where(better, dist, best_dist)
        have_best |= valid
    if not np.all(have_best):
        raise RuntimeError("Principal MC projection found no admissible candidate.")
    return selected


def _select_exact_torch(candidate_sq_distance: torch.Tensor, candidate_admissible: torch.Tensor, *, tol: float) -> torch.Tensor:
    n, k = candidate_sq_distance.shape
    selected = torch.zeros(n, dtype=torch.long, device=candidate_sq_distance.device)
    best_dist = torch.full((n,), float("inf"), dtype=candidate_sq_distance.dtype, device=candidate_sq_distance.device)
    have_best = torch.zeros(n, dtype=torch.bool, device=candidate_sq_distance.device)
    tie_tol = max(float(tol), 1.0e-15)
    tie_tol_t = torch.tensor(tie_tol, dtype=candidate_sq_distance.dtype, device=candidate_sq_distance.device)
    for i in range(k):
        dist = candidate_sq_distance[:, i]
        valid = candidate_admissible[:, i]
        better = valid & (~have_best | (dist < best_dist - tie_tol_t))
        selected = torch.where(better, torch.full_like(selected, i), selected)
        best_dist = torch.where(better, dist, best_dist)
        have_best |= valid
    if not bool(torch.all(have_best)):
        raise RuntimeError("Principal MC projection found no admissible candidate.")
    return selected


def principal_mc_projection_candidates(
    principal: np.ndarray,
    c_bar: np.ndarray | float,
    sin_phi: np.ndarray | float,
    *,
    sort_input: bool = True,
    tol: float = 1.0e-6,
) -> PrincipalProjectionDetails:
    """Return the five candidate projections and their feasibility diagnostics."""
    principal_arr = _coerce_principal_numpy(principal)
    principal_sorted = _sort_desc_numpy(principal_arr) if sort_input else principal_arr.copy()
    c_arr, s_arr = _broadcast_numpy(principal_sorted.shape[0], c_bar, sin_phi)
    candidate, sq_distance, yield_value, gap12, gap23, admissible = _candidate_arrays_numpy(
        principal_sorted,
        c_arr,
        s_arr,
        tol=tol,
    )
    selected_index = _select_exact_numpy(sq_distance, admissible, tol=tol)
    selected_label = np.asarray(PROJECTION_CANDIDATE_NAMES, dtype=object)[selected_index]
    projected = candidate[np.arange(candidate.shape[0]), selected_index]
    displacement_norm = np.linalg.norm(projected - principal_sorted, axis=1)
    return PrincipalProjectionDetails(
        principal_input=principal_arr,
        principal_sorted=principal_sorted,
        c_bar=c_arr,
        sin_phi=s_arr,
        projected_stress=projected,
        candidate_names=PROJECTION_CANDIDATE_NAMES,
        candidate_principal=candidate,
        feasible_mask=admissible,
        candidate_sq_distance=sq_distance,
        selected_index=selected_index,
        selected_label=selected_label,
        displacement_norm=displacement_norm,
    )


def principal_mc_projection_candidates_torch(
    principal: torch.Tensor,
    c_bar: torch.Tensor | float,
    sin_phi: torch.Tensor | float,
    *,
    sort_input: bool = True,
    tol: float = 1.0e-6,
) -> PrincipalProjectionDetails:
    """Torch version of the candidate-level principal-space projector."""
    principal_input = _coerce_principal_torch(principal)
    work_dtype = torch.float64 if principal_input.dtype != torch.float64 else principal_input.dtype
    principal_arr = principal_input.to(dtype=work_dtype)
    principal_sorted = _sort_desc_torch(principal_arr) if sort_input else principal_arr.clone()
    c_arr, s_arr = _broadcast_torch(
        principal_sorted.shape[0],
        c_bar,
        sin_phi,
        device=principal_sorted.device,
        dtype=work_dtype,
    )
    candidate, sq_distance, yield_value, gap12, gap23, admissible = _candidate_arrays_torch(
        principal_sorted,
        c_arr,
        s_arr,
        tol=tol,
    )
    selected_index = _select_exact_torch(sq_distance, admissible, tol=tol)
    rows = torch.arange(candidate.shape[0], device=candidate.device)
    projected = candidate[rows, selected_index]
    displacement_norm = torch.linalg.norm(projected - principal_sorted, dim=1)
    selected_label = np.asarray(PROJECTION_CANDIDATE_NAMES, dtype=object)[selected_index.detach().cpu().numpy()]
    return PrincipalProjectionDetails(
        principal_input=principal_input,
        principal_sorted=principal_sorted,
        c_bar=c_arr,
        sin_phi=s_arr,
        projected_stress=projected,
        candidate_names=PROJECTION_CANDIDATE_NAMES,
        candidate_principal=candidate,
        feasible_mask=admissible,
        candidate_sq_distance=sq_distance,
        selected_index=selected_index,
        selected_label=selected_label,
        displacement_norm=displacement_norm,
    )


def mc_projection_candidate_details(
    stress_principal: np.ndarray,
    *,
    c_bar: np.ndarray | float,
    sin_phi: np.ndarray | float,
    sort_input: bool = True,
    tol: float = 1.0e-6,
) -> PrincipalProjectionDetails:
    """Compatibility wrapper for exact candidate-level numpy diagnostics."""
    return principal_mc_projection_candidates(
        stress_principal,
        c_bar=c_bar,
        sin_phi=sin_phi,
        sort_input=sort_input,
        tol=tol,
    )


def _softmin_sq_distance_numpy(
    sq_distance: np.ndarray,
    feasible_mask: np.ndarray,
    principal_sorted: np.ndarray,
    c_bar: np.ndarray,
) -> np.ndarray:
    scale = (np.linalg.norm(principal_sorted, axis=1) + c_bar + 1.0) ** 2
    scale = np.maximum(scale, 1.0e-12)
    d2_norm = sq_distance / scale[:, None]
    return np.where(feasible_mask, d2_norm, np.inf)


def _softmin_sq_distance_torch(
    sq_distance: torch.Tensor,
    feasible_mask: torch.Tensor,
    principal_sorted: torch.Tensor,
    c_bar: torch.Tensor,
) -> torch.Tensor:
    scale = (torch.linalg.norm(principal_sorted, dim=1) + c_bar + 1.0) ** 2
    scale = torch.clamp(scale, min=1.0e-12)
    d2_norm = sq_distance / scale.unsqueeze(1)
    return torch.where(feasible_mask, d2_norm, torch.full_like(d2_norm, float("inf")))


def project_mc_principal_numpy(
    stress_principal: np.ndarray,
    *,
    c_bar: np.ndarray | float,
    sin_phi: np.ndarray | float,
    mode: str = "exact",
    tau: float = 0.05,
    return_details: bool = False,
    sort_input: bool = True,
    tol: float = 1.0e-6,
) -> np.ndarray | tuple[np.ndarray, PrincipalProjectionDetails]:
    """Project principal stress onto the admissible Mohr-Coulomb set."""
    details = principal_mc_projection_candidates(stress_principal, c_bar, sin_phi, sort_input=sort_input, tol=tol)
    if mode == "exact":
        out = details.projected_stress
        return (out, details) if return_details else out
    if mode == "softmin":
        temp = max(float(tau), 1.0e-12)
        finite_dist = _softmin_sq_distance_numpy(
            np.asarray(details.candidate_sq_distance, dtype=float),
            np.asarray(details.feasible_mask, dtype=bool),
            np.asarray(details.principal_sorted, dtype=float),
            np.asarray(details.c_bar, dtype=float),
        )
        min_dist = np.min(finite_dist, axis=1, keepdims=True)
        weights = np.exp(-(finite_dist - min_dist) / temp)
        weights = np.where(np.isfinite(finite_dist), weights, 0.0)
        weight_sum = np.sum(weights, axis=1, keepdims=True)
        weights = np.where(weight_sum > 0.0, weights / weight_sum, weights)
        out = np.sum(weights[:, :, None] * details.candidate_principal, axis=1)
        if return_details:
            details.soft_weights = weights
            details.projected_stress = out
            details.displacement_norm = np.linalg.norm(out - details.principal_sorted, axis=1)
            return out, details
        return out
    raise ValueError(f"Unknown projection mode {mode!r}.")


def project_mc_principal_torch(
    stress_principal: torch.Tensor,
    *,
    c_bar: torch.Tensor | float,
    sin_phi: torch.Tensor | float,
    mode: str = "exact",
    tau: float = 0.05,
    return_details: bool = False,
    sort_input: bool = True,
    tol: float = 1.0e-6,
) -> torch.Tensor | tuple[torch.Tensor, PrincipalProjectionDetails]:
    """Torch version of the admissible Mohr-Coulomb principal-space projector."""
    out_dtype = _coerce_principal_torch(stress_principal).dtype
    details = principal_mc_projection_candidates_torch(stress_principal, c_bar, sin_phi, sort_input=sort_input, tol=tol)
    if mode == "exact":
        out = details.projected_stress.to(dtype=out_dtype)
        if return_details:
            details.projected_stress = out
        return (out, details) if return_details else out
    if mode == "softmin":
        temp = max(float(tau), 1.0e-12)
        finite_dist = _softmin_sq_distance_torch(
            details.candidate_sq_distance,
            details.feasible_mask,
            details.principal_sorted,
            details.c_bar,
        )
        min_dist = torch.min(finite_dist, dim=1, keepdim=True).values
        weights = torch.exp(-(finite_dist - min_dist) / temp)
        weights = torch.where(torch.isfinite(finite_dist), weights, torch.zeros_like(weights))
        weight_sum = torch.sum(weights, dim=1, keepdim=True)
        weights = torch.where(weight_sum > 0.0, weights / weight_sum, weights)
        out = torch.sum(weights.unsqueeze(-1) * details.candidate_principal, dim=1).to(dtype=out_dtype)
        if return_details:
            details.soft_weights = weights
            details.projected_stress = out
            details.displacement_norm = torch.linalg.norm(out.to(dtype=details.principal_sorted.dtype) - details.principal_sorted, dim=1)
            return out, details
        return out
    raise ValueError(f"Unknown projection mode {mode!r}.")


def project_principal_mc(
    stress_principal: np.ndarray,
    c_bar: np.ndarray | float,
    sin_phi: np.ndarray | float,
    *,
    mode: str = "exact",
    temperature: float = 0.05,
    sort_input: bool = True,
    tol: float = 1.0e-6,
    return_details: bool = False,
) -> np.ndarray | tuple[np.ndarray, PrincipalProjectionDetails]:
    """Backward-compatible alias for `project_mc_principal_numpy`."""
    return project_mc_principal_numpy(
        stress_principal,
        c_bar=c_bar,
        sin_phi=sin_phi,
        mode=mode,
        tau=temperature,
        sort_input=sort_input,
        tol=tol,
        return_details=return_details,
    )


def project_principal_mc_torch(
    stress_principal: torch.Tensor,
    c_bar: torch.Tensor | float,
    sin_phi: torch.Tensor | float,
    *,
    mode: str = "exact",
    temperature: float = 0.05,
    sort_input: bool = True,
    tol: float = 1.0e-6,
    return_details: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, PrincipalProjectionDetails]:
    """Backward-compatible alias for `project_mc_principal_torch`."""
    return project_mc_principal_torch(
        stress_principal,
        c_bar=c_bar,
        sin_phi=sin_phi,
        mode=mode,
        tau=temperature,
        sort_input=sort_input,
        tol=tol,
        return_details=return_details,
    )
