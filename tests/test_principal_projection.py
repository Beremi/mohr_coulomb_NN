import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mc_surrogate.principal_projection import (
    PROJECTION_CANDIDATE_NAMES,
    project_mc_principal_numpy,
    project_mc_principal_torch,
    principal_mc_projection_candidates,
)


def _yield_function(principal: np.ndarray, c_bar: np.ndarray | float, sin_phi: np.ndarray | float) -> np.ndarray:
    sigma = np.asarray(principal, dtype=float)
    if sigma.ndim == 1:
        sigma = sigma[None, :]
    c = np.asarray(c_bar, dtype=float).reshape(-1)
    s = np.asarray(sin_phi, dtype=float).reshape(-1)
    if c.size == 1:
        c = np.full(sigma.shape[0], c.item(), dtype=float)
    if s.size == 1:
        s = np.full(sigma.shape[0], s.item(), dtype=float)
    return (1.0 + s) * sigma[:, 0] - (1.0 - s) * sigma[:, 2] - c


def _assert_admissible(principal: np.ndarray, c_bar: np.ndarray | float, sin_phi: np.ndarray | float, *, tol: float = 1.0e-10) -> None:
    sigma = np.asarray(principal, dtype=float)
    if sigma.ndim == 1:
        sigma = sigma[None, :]
    assert np.all(sigma[:, 0] >= sigma[:, 1] - tol)
    assert np.all(sigma[:, 1] >= sigma[:, 2] - tol)
    assert np.all(_yield_function(sigma, c_bar, sin_phi) <= tol)


def test_exact_projection_identity_inside_admissible_set():
    stress = np.array([2.0, 1.0, 0.0])
    projected, details = project_mc_principal_numpy(
        stress,
        c_bar=4.0,
        sin_phi=0.5,
        return_details=True,
    )

    expected = np.array([[2.0, 1.0, 0.0]])
    assert np.allclose(projected, expected)
    assert np.allclose(details.projected_stress, expected)
    assert details.selected_index.tolist() == [0]
    assert details.selected_label.tolist() == ["pass_through"]
    _assert_admissible(projected, 4.0, 0.5)


@pytest.mark.parametrize(
    "label, stress, expected, selected_index",
    [
        ("smooth", np.array([17.0 / 3.0, 1.0, -1.0]), np.array([[8.0 / 3.0, 1.0, 0.0]]), 1),
        ("left_edge", np.array([5.0, 4.0, 0.0]), np.array([[3.0, 3.0, 1.0]]), 2),
        ("right_edge", np.array([10.5, 0.0, -0.5]), np.array([[3.0, 1.0, 1.0]]), 3),
        ("apex", np.array([8.0, 7.0, 6.0]), np.array([[4.0, 4.0, 4.0]]), 4),
    ],
)
def test_exact_projection_matches_branch_examples(label, stress, expected, selected_index):
    projected, details = project_mc_principal_numpy(
        stress,
        c_bar=4.0,
        sin_phi=0.5,
        return_details=True,
    )

    assert np.allclose(projected, expected)
    assert np.allclose(details.projected_stress, expected)
    assert details.selected_index.tolist() == [selected_index]
    assert details.selected_label.tolist() == [label]
    assert PROJECTION_CANDIDATE_NAMES[selected_index] == label
    _assert_admissible(projected, 4.0, 0.5)


def test_unsorted_input_is_canonicalized_before_projection():
    sorted_input = np.array([[17.0 / 3.0, 1.0, -1.0]])
    unsorted_input = np.array([1.0, 17.0 / 3.0, -1.0])

    projected_sorted = project_mc_principal_numpy(sorted_input, c_bar=4.0, sin_phi=0.5)
    projected_unsorted, details = project_mc_principal_numpy(
        unsorted_input,
        c_bar=4.0,
        sin_phi=0.5,
        return_details=True,
    )

    assert np.allclose(projected_unsorted, projected_sorted)
    assert np.allclose(details.principal_sorted, sorted_input)
    assert details.selected_label.tolist() == ["smooth"]


def test_candidate_details_and_admissibility_are_consistent():
    rng = np.random.default_rng(0)
    stress = rng.normal(size=(12, 3)) * 4.0
    c_bar = np.linspace(3.5, 5.0, stress.shape[0])
    sin_phi = np.linspace(0.2, 0.7, stress.shape[0])

    projected, details = project_mc_principal_numpy(
        stress,
        c_bar=c_bar,
        sin_phi=sin_phi,
        return_details=True,
    )

    rows = np.arange(stress.shape[0])
    selected = details.candidate_principal[rows, details.selected_index]
    assert np.allclose(projected, selected)
    assert np.allclose(details.projected_stress, projected)
    assert np.allclose(details.displacement_norm, np.linalg.norm(projected - details.principal_sorted, axis=1))
    assert np.all(details.feasible_mask.sum(axis=1) >= 1)
    _assert_admissible(projected, c_bar, sin_phi)


def test_numpy_torch_parity_exact_and_softmin():
    rng = np.random.default_rng(1)
    stress = rng.normal(size=(10, 3)) * 3.0
    c_bar = np.linspace(3.8, 5.1, stress.shape[0])
    sin_phi = np.linspace(0.15, 0.65, stress.shape[0])

    for mode in ("exact", "softmin"):
        np_proj = project_mc_principal_numpy(stress, c_bar=c_bar, sin_phi=sin_phi, mode=mode, tau=0.07)
        torch_proj = project_mc_principal_torch(
            torch.tensor(stress, dtype=torch.float64),
            c_bar=torch.tensor(c_bar, dtype=torch.float64),
            sin_phi=torch.tensor(sin_phi, dtype=torch.float64),
            mode=mode,
            tau=0.07,
        )

        assert np.allclose(np_proj, torch_proj.detach().cpu().numpy(), atol=1.0e-10, rtol=1.0e-8)


def test_exact_torch_projection_has_finite_gradients_away_from_ties():
    stress = torch.tensor([[5.0, 4.0, 0.0]], dtype=torch.float64, requires_grad=True)
    projected = project_mc_principal_torch(stress, c_bar=4.0, sin_phi=0.5)
    projected.sum().backward()

    assert torch.isfinite(projected).all()
    assert torch.isfinite(stress.grad).all()


def test_softmin_projection_remains_admissible():
    rng = np.random.default_rng(2)
    stress = rng.normal(size=(16, 3)) * 5.0
    c_bar = np.linspace(3.9, 5.2, stress.shape[0])
    sin_phi = np.linspace(0.1, 0.75, stress.shape[0])

    projected, details = project_mc_principal_numpy(
        stress,
        c_bar=c_bar,
        sin_phi=sin_phi,
        mode="softmin",
        tau=0.08,
        return_details=True,
    )

    assert np.all(details.soft_weights >= 0.0)
    assert np.allclose(np.sum(details.soft_weights, axis=1), 1.0)
    _assert_admissible(projected, c_bar, sin_phi, tol=1.0e-9)

