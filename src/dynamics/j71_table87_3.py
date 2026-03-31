from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class J71DensityPolySection:
    """
    Piecewise polynomial section for MicroCosm Table 8.7-3 total density fit.

    P1(h, T) = log10(rho_DT [g/cm^3])
             = sum_i sum_j a_ij * h^i * T^j
    """
    name: str
    h_min_km: float
    h_max_km: float
    t_min_k: float
    t_max_k: float
    coeffs: Tuple[Tuple[float, ...], ...]


DENSITY_POLY_SECTIONS_87_3: Tuple[J71DensityPolySection, ...] = (
    J71DensityPolySection(
        name="90-200km_500-1900K",
        h_min_km=90.0,
        h_max_km=200.0,
        t_min_k=500.0,
        t_max_k=1900.0,
        coeffs=(
            (4.22085,     0.98393e-2,  -0.64952e-5,   0.14715e-8),
            (-0.20134,   -0.23412e-3,   0.15337e-6,  -0.34675e-10),
            (0.78592e-3,  0.16966e-5,  -0.22060e-8,   0.25007e-12),
            (-0.12087e-5, -0.34360e-8,  0.22457e-11, -0.51069e-15),
        ),
    ),
    J71DensityPolySection(
        name="200-500km_500-800K",
        h_min_km=200.0,
        h_max_km=500.0,
        t_min_k=500.0,
        t_max_k=800.0,
        coeffs=(
            (-0.12838e+2,  0.40709e-2,  0.97074e-5,  -0.10643e-7),
            ( 0.82282e-1, -0.31215e-3,  0.26543e-6,  -0.55193e-10),
            (-0.68951e-3,  0.24402e-5, -0.27058e-8,   0.99003e-12),
            ( 0.11263e-5, -0.41807e-8,  0.50617e-11, -0.20484e-14),
        ),
    ),
    J71DensityPolySection(
        name="200-500km_800-1900K",
        h_min_km=200.0,
        h_max_km=500.0,
        t_min_k=800.0,
        t_max_k=1900.0,
        coeffs=(
            (-8.45950,    -0.15000e-3, -0.62640e-6,   0.24612e-9),
            (-0.28395e-1,  0.17760e-6,  0.61398e-8,  -0.23362e-11),
            ( 0.55998e-5,  0.77461e-7, -0.59492e-10,  0.14921e-13),
            ( 0.39434e-8, -0.76435e-10, 0.58333e-13, -0.14595e-16),
        ),
    ),
    J71DensityPolySection(
        name="500-1000km_500-800K",
        h_min_km=500.0,
        h_max_km=1000.0,
        t_min_k=500.0,
        t_max_k=800.0,
        coeffs=(
            ( 0.77659e+2,  0.16727,    -0.56570e-4,  -0.50424e-7),
            ( 0.30638,    -0.98936e-3,  0.74932e-6,  -0.53178e-10),
            ( 0.38935e-3,  0.12973e-5, -0.19776e-8,   0.14191e-12),
            ( 0.15962e-6, -0.54049e-9,  0.46709e-12, -0.71886e-16),
        ),
    ),
    J71DensityPolySection(
        name="500-1000km_800-1900K",
        h_min_km=500.0,
        h_max_km=1000.0,
        t_min_k=800.0,
        t_max_k=1900.0,
        coeffs=(
            ( 0.50081e+2, -0.12600,     0.83896e-4,  -0.18276e-7),
            (-0.30572,     0.61706e-3, -0.41443e-6,   0.91096e-10),
            ( 0.41767e-3, -0.88743e-6,  0.61040e-9,  -0.13634e-12),
            (-0.17965e-6,  0.39386e-9, -0.27639e-12,  0.62649e-16),
        ),
    ),
    J71DensityPolySection(
        name="1000-2500km_500-800K",
        h_min_km=1000.0,
        h_max_km=2500.0,
        t_min_k=500.0,
        t_max_k=800.0,
        coeffs=(
            ( 0.365352e+2, -0.26156,     0.41963e-3,  -0.21661e-6),
            (-0.48352e-1,  0.26801e-3,  -0.48214e-6,   0.27095e-9),
            ( 0.11141e-4, -0.7750e-7,    0.16042e-9,  -0.99055e-13),
            (-0.25059e-10, 0.44725e-11, -0.14085e-13,  0.10443e-16),
        ),
    ),
    J71DensityPolySection(
        name="1000-2500km_800-1900K",
        h_min_km=1000.0,
        h_max_km=2500.0,
        t_min_k=800.0,
        t_max_k=1900.0,
        coeffs=(
            ( 0.52410e+2, -0.20652,     0.21642e-3,  -0.90623e-7,   0.13054e-10),
            (-0.14355,     0.43113e-3, -0.46137e-6,   0.20179e-9,  -0.30888e-13),
            ( 0.87693e-4, -0.27157e-6,  0.29745e-9,  -0.13425e-12,  0.21370e-16),
            (-0.15716e-7,  0.49631e-10, -0.55297e-13,  0.25432e-16, -0.41304e-20),
        ),
    ),
)