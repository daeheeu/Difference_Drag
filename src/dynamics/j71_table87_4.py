from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class J71HeliumPolySection:
    """
    Piecewise polynomial section for MicroCosm Table 8.7-4 helium fit.

    Q1(h, T) = log10(n_He [#/cm^3])
             = sum_i sum_j b_ij * h^i * T^j
    """
    name: str
    h_min_km: float
    h_max_km: float
    t_min_k: float
    t_max_k: float
    coeffs: Tuple[Tuple[float, ...], ...]


HELIUM_POLY_SECTIONS_87_4: Tuple[J71HeliumPolySection, ...] = (
    J71HeliumPolySection(
        name="500-1000km_500-800K",
        h_min_km=500.0,
        h_max_km=1000.0,
        t_min_k=500.0,
        t_max_k=800.0,
        coeffs=(
            (9.37120,     -0.52634e-2,  0.52983e-5,  -0.20471e-8),
            (-0.13141e-1,  0.31218e-4, -0.32598e-7,   0.12573e-10),
            (0.26071e-5,  -0.75730e-8,  0.93058e-11, -0.40669e-14),
            (-0.52156e-9,  0.19056e-11, -0.26578e-14, 0.12535e-17),
        ),
    ),
    J71HeliumPolySection(
        name="500-1000km_800-1900K",
        h_min_km=500.0,
        h_max_km=1000.0,
        t_min_k=800.0,
        t_max_k=1900.0,
        coeffs=(
            (8.39140,     -0.16433e-2,  0.78032e-6,  -0.14323e-9),
            (-0.69049e-2,  0.84138e-5, -0.44577e-8,   0.85627e-12),
            (0.15893e-5,  -0.35863e-8,  0.35476e-11, -0.12985e-14),
            (-0.11829e-9,  0.26138e-12, -0.25227e-15, 0.89714e-19),
        ),
    ),
    J71HeliumPolySection(
        name="1000-2500km_500-800K",
        h_min_km=1000.0,
        h_max_km=2500.0,
        t_min_k=500.0,
        t_max_k=800.0,
        coeffs=(
            (9.10450,     -4.3410e-2,   0.40292e-5,  -0.14522e-8),
            (-0.12259e-1,  0.27951e-4, -0.27972e-7,   0.10371e-10),
            (0.15893e-5,  -0.35863e-8,  0.35476e-11, -0.12985e-14),
            (-0.11829e-9,  0.26138e-12, -0.25227e-15, 0.89714e-19),
        ),
    ),
    J71HeliumPolySection(
        name="1000-2500km_800-1900K",
        h_min_km=1000.0,
        h_max_km=2500.0,
        t_min_k=800.0,
        t_max_k=1900.0,
        coeffs=(
            (8.61200,     -0.25363e-2,  0.18979e-5,  -0.73696e-9,   0.11388e-12),
            (-0.84847e-2,  0.14084e-4, -0.11386e-7,   0.44871e-11, -0.69064e-15),
            (0.11543e-5,  -0.19884e-8,  0.16635e-11, -0.67628e-15,  0.10706e-19),
            (-0.94521e-10, 0.17387e-12, -0.15368e-15, 0.65402e-19, -0.10760e-22),
        ),
    ),
)