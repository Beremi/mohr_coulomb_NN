"""Mohr-Coulomb constitutive surrogate toolkit."""

from .materials import davis_reduction, isotropic_moduli_from_young_poisson
from .mohr_coulomb import BRANCH_NAMES, constitutive_update_3d

try:  # Optional dependency chain pulls in h5py.
    from .inference import ConstitutiveSurrogate
except Exception:  # pragma: no cover - import-time fallback for lean environments.
    ConstitutiveSurrogate = None

__all__ = [
    "BRANCH_NAMES",
    "ConstitutiveSurrogate",
    "constitutive_update_3d",
    "davis_reduction",
    "isotropic_moduli_from_young_poisson",
]
