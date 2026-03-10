import numpy as np

from mc_surrogate.materials import build_reduced_material_from_raw
from mc_surrogate.mohr_coulomb import BRANCH_TO_ID, constitutive_update_3d
from mc_surrogate.voigt import stress_voigt_to_tensor, tensor_to_strain_voigt


def _default_material(n=1):
    return build_reduced_material_from_raw(
        c0=np.full(n, 15.0),
        phi_rad=np.deg2rad(np.full(n, 35.0)),
        psi_rad=np.deg2rad(np.full(n, 0.0)),
        young=np.full(n, 2.0e4),
        poisson=np.full(n, 0.3),
        strength_reduction=np.full(n, 1.2),
        davis_type=["B"] * n,
    )


def test_elastic_hydrostatic_response():
    reduced = _default_material(1)
    strain = np.array([[1.0e-5, 1.0e-5, 1.0e-5, 0.0, 0.0, 0.0]])
    result = constitutive_update_3d(
        strain,
        c_bar=reduced.c_bar,
        sin_phi=reduced.sin_phi,
        shear=reduced.shear,
        bulk=reduced.bulk,
        lame=reduced.lame,
    )
    expected = 3.0 * reduced.bulk[0] * 1.0e-5
    assert result.branch_id[0] == BRANCH_TO_ID["elastic"]
    assert np.allclose(result.stress[0, :3], expected, rtol=1e-8, atol=1e-10)
    assert np.allclose(result.stress[0, 3:], 0.0, atol=1e-12)


def test_rotational_invariance():
    reduced = _default_material(1)
    principal = np.array([[2.0e-4, 0.5e-4, -1.0e-4]])
    strain_diag = tensor_to_strain_voigt(np.diag(principal[0])[None, :, :])

    theta = 0.37
    rot = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    strain_rot = tensor_to_strain_voigt((rot @ np.diag(principal[0]) @ rot.T)[None, :, :])

    base = constitutive_update_3d(
        strain_diag,
        c_bar=reduced.c_bar,
        sin_phi=reduced.sin_phi,
        shear=reduced.shear,
        bulk=reduced.bulk,
        lame=reduced.lame,
    )
    turned = constitutive_update_3d(
        strain_rot,
        c_bar=reduced.c_bar,
        sin_phi=reduced.sin_phi,
        shear=reduced.shear,
        bulk=reduced.bulk,
        lame=reduced.lame,
    )
    sigma_base = stress_voigt_to_tensor(base.stress)[0]
    sigma_turn = stress_voigt_to_tensor(turned.stress)[0]
    sigma_expected = rot @ sigma_base @ rot.T
    assert np.allclose(sigma_turn, sigma_expected, atol=1e-8)


def test_apex_branch_under_strong_hydrostatic_compression():
    reduced = _default_material(1)
    strain = np.array([[5.0e-2, 5.0e-2, 5.0e-2, 0.0, 0.0, 0.0]])
    result = constitutive_update_3d(
        strain,
        c_bar=reduced.c_bar,
        sin_phi=reduced.sin_phi,
        shear=reduced.shear,
        bulk=reduced.bulk,
        lame=reduced.lame,
    )
    assert result.branch_id[0] == BRANCH_TO_ID["apex"]
    assert np.allclose(result.stress[0, 0], result.stress[0, 1])
    assert np.allclose(result.stress[0, 1], result.stress[0, 2])
