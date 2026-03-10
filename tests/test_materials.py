import numpy as np

from mc_surrogate.materials import davis_reduction, isotropic_moduli_from_young_poisson


def test_isotropic_moduli_positive():
    shear, bulk, lame = isotropic_moduli_from_young_poisson(np.array([1.0e4]), np.array([0.3]))
    assert shear.shape == (1,)
    assert bulk.shape == (1,)
    assert lame.shape == (1,)
    assert shear[0] > 0.0
    assert bulk[0] > 0.0
    assert lame[0] > 0.0


def test_davis_reduction_shapes_and_bounds():
    c_bar, sin_phi = davis_reduction(
        c0=np.array([10.0, 15.0, 20.0]),
        phi_rad=np.deg2rad(np.array([30.0, 35.0, 40.0])),
        psi_rad=np.deg2rad(np.array([0.0, 5.0, 10.0])),
        strength_reduction=np.array([1.0, 1.5, 2.0]),
        davis_type=["A", "B", "C"],
    )
    assert c_bar.shape == (3,)
    assert sin_phi.shape == (3,)
    assert np.all(c_bar > 0.0)
    assert np.all(np.abs(sin_phi) < 1.0)
