from __future__ import annotations

"""Orbit Determination (OD) pipeline.

This module is a refactor-friendly extraction of the OD portion of
`src/cases/case_c_od_op_navsol.py`.

Design goals
- Works with your existing Case-C style config JSON (it will ignore OP keys).
- Produces self-contained `od_solution.json` that can drive OP without needing
  the original config.
- Writes `od_fit.csv` and `od_summary.json` for quick diagnostics.

Typical usage
    python -m src.pipelines.od configs/case_c_od_op_only_2025.json

Or call programmatically
    from src.pipelines.od import run_od
    summary = run_od(cfg_dict)
"""

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import datetime as dt

import numpy as np
import pandas as pd

from src.orekit_bootstrap import init_orekit

from src.dynamics.force_model import (
    ForceModelCfg,
    apply_force_models,
    build_force_model_bundle,
    force_cfg_from_dict,
    force_cfg_to_dict,
)


# -----------------------------
# Config dataclasses
# -----------------------------


@dataclass
class ODWindow:
    """Which portion of navsol to use for OD."""

    # Arc selection
    arc_index: Optional[int] = None
    arc_indices: Optional[List[int]] = None
    arc_mode: Optional[str] = None  # all | longest | longest_n | min_duration

    longest_n: int = 2
    min_arc_s: float = 600.0
    anchor: str = "last"  # first | last | longest

    # Optional explicit time window
    time_start_utc: Optional[str] = None
    time_end_utc: Optional[str] = None

    # OD measurement processing
    downsample_s: int = 1
    pos_sigma_m: float = 10.0


@dataclass
class ODRunCfg:
    """Top-level OD run config."""

    navsol_csv: str
    arc_gap_s: float = 1.5

    od: ODWindow = None
    forces: ForceModelCfg = None

    outputs_dir: str = "outputs"

    # Optional
    orekit_data_path: Optional[str] = None
    orekit_extra_data_paths: Optional[List[str]] = None
    reference_frame: str = "EME2000"

    # Solver knobs
    max_iterations: int = 300
    max_evaluations: int = 8000


# -----------------------------
# NavSol loading / arc handling
# -----------------------------


def _as_dt_utc(s: str) -> pd.Timestamp:
    return pd.to_datetime(s, utc=True)


def load_navsol_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    need = {"iso_utc", "x_m", "y_m", "z_m"}
    if not need.issubset(df.columns):
        raise ValueError(f"NavSol CSV must contain columns: {sorted(need)}")

    df["dt"] = pd.to_datetime(df["iso_utc"], utc=True)
    df = df.sort_values("dt").reset_index(drop=True)
    return df


def load_navsol(path: str | Path) -> pd.DataFrame:
    """Load NavSol from CSV or .navSol text.

    This is kept compatible with your existing case script.
    """

    p = Path(path)

    # 1) direct CSV
    if p.exists() and p.suffix.lower() == ".csv":
        return load_navsol_csv(p)

    # 2) search variations
    candidates: List[Path] = []
    candidates.append(p)
    candidates.append(p.with_suffix(".navSol"))
    candidates.append(p.with_suffix(".navsol"))
    candidates.append(p.with_name(p.name.replace("_", "-")))
    candidates.append(p.with_name(p.name.replace("_", "-")).with_suffix(".navSol"))
    candidates.append(p.with_name(p.name.replace("_", "-")).with_suffix(".navsol"))

    found: Optional[Path] = None
    for c in candidates:
        if c.exists():
            found = c
            break

    if found is None:
        raise FileNotFoundError(
            f"NavSol input not found: {p} (also tried .navSol/.navsol and '_'↔'-')"
        )

    if found.suffix.lower() == ".csv":
        return load_navsol_csv(found)

    # ---- parse .navSol text ----
    rows: List[Dict[str, Any]] = []
    with open(found, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            toks = line.split()
            # Expect: 001 YYYY MM DD HH MM SSmmm valid mode x y z 0 nsv prn...
            if len(toks) < 14 or toks[0] != "001":
                continue

            year, mon, day = int(toks[1]), int(toks[2]), int(toks[3])
            hh, mm = int(toks[4]), int(toks[5])
            ssmmm = int(toks[6])
            sec = ssmmm // 1000
            msec = ssmmm % 1000

            valid = int(toks[7])
            mode = int(toks[8])
            x, y, z = float(toks[9]), float(toks[10]), float(toks[11])
            nsv = int(toks[13])
            prns = toks[14 : 14 + nsv]

            ts = dt.datetime(
                year, mon, day, hh, mm, sec, msec * 1000, tzinfo=dt.timezone.utc
            )
            iso = ts.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

            rows.append(
                {
                    "iso_utc": iso,
                    "x_m": x,
                    "y_m": y,
                    "z_m": z,
                    "valid": valid,
                    "mode": mode,
                    "nsv": nsv,
                    "prns": ",".join(prns),
                }
            )

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise ValueError(f"No valid records parsed from {found}")

    df["dt"] = pd.to_datetime(df["iso_utc"], utc=True)
    df = df.sort_values("dt").reset_index(drop=True)
    return df


def split_arcs(df: pd.DataFrame, gap_s: float) -> List[Tuple[int, int]]:
    t = df["dt"].astype("int64").to_numpy() / 1e9
    gaps = np.diff(t)
    cut = np.where(gaps > gap_s)[0]
    starts = [0] + (cut + 1).tolist()
    ends = cut.tolist() + [len(df) - 1]
    return list(zip(starts, ends))


def select_od_segment(df: pd.DataFrame, gap_s: float, od: ODWindow) -> pd.DataFrame:
    """Select OD measurement rows according to ODWindow."""

    # Explicit time window
    if od.time_start_utc and od.time_end_utc:
        ts = _as_dt_utc(od.time_start_utc)
        te = _as_dt_utc(od.time_end_utc)
        seg = df[(df["dt"] >= ts) & (df["dt"] <= te)].copy()
        if len(seg) < 5:
            raise ValueError("OD time window too small or no data in window.")
        return seg.reset_index(drop=True)

    arcs = split_arcs(df, gap_s)

    # Explicit arc indices
    if od.arc_indices:
        chunks = []
        for ai in od.arc_indices:
            if ai < 0 or ai >= len(arcs):
                raise IndexError(
                    f"od.arc_indices contains out-of-range arc: {ai} (arcs={len(arcs)})"
                )
            s, e = arcs[ai]
            chunks.append(df.iloc[s : e + 1])
        return pd.concat(chunks).sort_values("dt").reset_index(drop=True)

    # arc_mode auto selection
    if od.arc_mode:
        arc_mode = od.arc_mode.strip().lower()

        meta: List[Tuple[int, int, int, float]] = []
        for i, (s, e) in enumerate(arcs):
            t0 = df.loc[s, "dt"]
            t1 = df.loc[e, "dt"]
            dur = (t1 - t0).total_seconds()
            meta.append((i, s, e, dur))

        if arc_mode == "all":
            return df.copy().reset_index(drop=True)

        if arc_mode == "longest":
            _i, s, e, _dur = max(meta, key=lambda x: x[3])
            return df.iloc[s : e + 1].copy().reset_index(drop=True)

        if arc_mode == "longest_n":
            n = int(od.longest_n)
            pick = sorted(meta, key=lambda x: x[3], reverse=True)[:n]
            chunks = [df.iloc[s : e + 1] for (_i, s, e, _dur) in sorted(pick, key=lambda x: x[0])]
            return pd.concat(chunks).sort_values("dt").reset_index(drop=True)

        if arc_mode == "min_duration":
            min_arc_s = float(od.min_arc_s)
            pick = [x for x in meta if x[3] >= min_arc_s]
            if not pick:
                raise ValueError(f"No arcs with duration >= {min_arc_s}s")
            chunks = [df.iloc[s : e + 1] for (_i, s, e, _dur) in pick]
            return pd.concat(chunks).sort_values("dt").reset_index(drop=True)

        raise ValueError(f"Unknown od.arc_mode: {od.arc_mode}")

    # single arc index
    if od.arc_index is None:
        raise ValueError(
            "OD selection requires arc_index, arc_indices, arc_mode, or time window."
        )
    if od.arc_index < 0 or od.arc_index >= len(arcs):
        raise IndexError(f"od.arc_index out of range: {od.arc_index} (arcs={len(arcs)})")

    s, e = arcs[od.arc_index]
    return df.iloc[s : e + 1].copy().reset_index(drop=True)


def downsample(df: pd.DataFrame, step_s: int) -> pd.DataFrame:
    if step_s <= 1:
        return df
    t0 = df["dt"].iloc[0]
    dt_s = (df["dt"] - t0).dt.total_seconds()
    mask = (dt_s % step_s) < 1e-6
    return df.loc[mask].copy().reset_index(drop=True)


def pick_reference_index(df: pd.DataFrame, gap_s: float, anchor: str) -> int:
    """Pick reference epoch index within the chosen OD segment."""

    arcs = split_arcs(df, gap_s)
    if not arcs:
        return 0

    anchor = (anchor or "last").lower()

    if anchor == "first":
        return arcs[0][0]

    if anchor == "last":
        return arcs[-1][0]

    if anchor == "longest":
        best = None
        for (s, e) in arcs:
            dur = (df.loc[e, "dt"] - df.loc[s, "dt"]).total_seconds()
            if best is None or dur > best[0]:
                best = (dur, s)
        return int(best[1])

    return arcs[-1][0]


def estimate_v_ecef(df: pd.DataFrame, idx0: int) -> np.ndarray:
    """Estimate ECEF velocity around idx0.

    - Prefer a 3-point centered-ish finite difference (idx0 and idx0+2) for stability.
    - Fallback to 2-point if near the end.
    """

    if len(df) < 2:
        raise ValueError("Need >=2 points to estimate velocity")

    if idx0 < 0:
        idx0 = 0

    if idx0 + 2 < len(df):
        r0 = df.loc[idx0, ["x_m", "y_m", "z_m"]].to_numpy(dtype=float)
        r2 = df.loc[idx0 + 2, ["x_m", "y_m", "z_m"]].to_numpy(dtype=float)
        t0 = df.loc[idx0, "dt"].value / 1e9
        t2 = df.loc[idx0 + 2, "dt"].value / 1e9
        return (r2 - r0) / (t2 - t0)

    # Fallback: last two points
    r_prev = df.loc[len(df) - 2, ["x_m", "y_m", "z_m"]].to_numpy(dtype=float)
    r_last = df.loc[len(df) - 1, ["x_m", "y_m", "z_m"]].to_numpy(dtype=float)
    t_prev = df.loc[len(df) - 2, "dt"].value / 1e9
    t_last = df.loc[len(df) - 1, "dt"].value / 1e9
    return (r_last - r_prev) / (t_last - t_prev)


# -----------------------------
# OD pipeline
# -----------------------------


def _parse_run_cfg(cfg_raw: Dict[str, Any]) -> ODRunCfg:
    """Parse either a dedicated OD config or an existing Case-C config."""

    # Inputs: support both new and legacy naming
    inputs = cfg_raw.get("inputs", {})
    navsol_csv = (
        inputs.get("navsol_csv")
        or inputs.get("navsol_day1_csv")
        or cfg_raw.get("navsol_csv")
    )

    if not navsol_csv:
        raise KeyError(
            "Missing navsol input. Expected one of: inputs.navsol_csv, inputs.navsol_day1_csv, navsol_csv"
        )

    arc_gap_s = float(cfg_raw.get("arc_gap_s", cfg_raw.get("arc_gap", 1.5)))

    od = ODWindow(**cfg_raw.get("od", {}))
    forces = force_cfg_from_dict(cfg_raw.get("forces", {}))

    outputs_dir = str(cfg_raw.get("outputs_dir", "outputs"))
    orekit_data_path = cfg_raw.get("orekit_data_path")
    orekit_extra_data_paths = cfg_raw.get("orekit_extra_data_paths")
    reference_frame = str(cfg_raw.get("reference_frame", "EME2000"))

    max_iterations = int(cfg_raw.get("max_iterations", 300))
    max_evaluations = int(cfg_raw.get("max_evaluations", 8000))

    return ODRunCfg(
        navsol_csv=navsol_csv,
        arc_gap_s=arc_gap_s,
        od=od,
        forces=forces,
        outputs_dir=outputs_dir,
        orekit_data_path=orekit_data_path,
        orekit_extra_data_paths=orekit_extra_data_paths,
        reference_frame=reference_frame,
        max_iterations=max_iterations,
        max_evaluations=max_evaluations,
    )


def run_od(cfg_raw: Dict[str, Any]) -> Dict[str, Any]:
    """Run OD and write artifacts.

    Returns a JSON-serializable summary dict.
    """

    cfg = _parse_run_cfg(cfg_raw)

    out_dir = Path(cfg.outputs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # JVM + orekit-data
    init_orekit(
        cfg.orekit_data_path,
        extra_data_paths=cfg.orekit_extra_data_paths,
    )

    # Orekit imports AFTER init_orekit()
    from org.hipparchus.optim.nonlinear.vector.leastsquares import LevenbergMarquardtOptimizer
    from org.orekit.time import AbsoluteDate, TimeScalesFactory
    from org.orekit.frames import FramesFactory
    from org.orekit.utils import IERSConventions, Constants
    from org.orekit.utils import PVCoordinates
    from org.orekit.orbits import CartesianOrbit, PositionAngleType
    from org.orekit.propagation.conversion import DormandPrince853IntegratorBuilder
    from org.orekit.propagation.conversion import NumericalPropagatorBuilder
    from org.orekit.bodies import OneAxisEllipsoid
    from org.orekit.estimation.leastsquares import BatchLSEstimator
    from org.orekit.estimation.measurements import ObservableSatellite, Position
    from org.hipparchus.geometry.euclidean.threed import Vector3D
    from org.orekit.forces.drag import DragSensitive
    from org.orekit.forces.radiation import RadiationSensitive

    def _get_reference_inertial_frame(name: str):
        key = str(name or "EME2000").strip().upper()

        if key == "EME2000":
            return FramesFactory.getEME2000(), "EME2000"
        if key == "TOD":
            return FramesFactory.getTOD(IERSConventions.IERS_2010, True), "TOD"
        
        raise ValueError(f"Unsupported reference_frame: {name}")


    # Load navsol
    df = load_navsol(cfg.navsol_csv)

    # Select OD segment
    od_df_full = select_od_segment(df, cfg.arc_gap_s, cfg.od)
    od_df = downsample(od_df_full, int(cfg.od.downsample_s))

    if len(od_df) < 5:
        raise ValueError("OD segment too small after downsample (need >=5 points).")

    # Frames
    utc = TimeScalesFactory.getUTC()
    inertial, inertial_name = _get_reference_inertial_frame(cfg.reference_frame)
    itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)

    earth = OneAxisEllipsoid(
        Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
        Constants.WGS84_EARTH_FLATTENING,
        itrf,
    )

    def to_absdate(ts: pd.Timestamp) -> AbsoluteDate:
        return AbsoluteDate(
            ts.year,
            ts.month,
            ts.day,
            ts.hour,
            ts.minute,
            ts.second + ts.microsecond * 1e-6,
            utc,
        )

    # Initial guess at reference epoch
    ref_idx_full = pick_reference_index(od_df_full, cfg.arc_gap_s, cfg.od.anchor)
    # Make sure we can estimate v
    ref_idx_full = max(0, min(int(ref_idx_full), len(od_df_full) - 2))

    r0_ecef = od_df_full.loc[ref_idx_full, ["x_m", "y_m", "z_m"]].to_numpy(dtype=float)
    v0_ecef = estimate_v_ecef(od_df_full, ref_idx_full)

    t_ref_ts = pd.to_datetime(od_df_full.loc[ref_idx_full, "dt"], utc=True)
    t_ref = to_absdate(t_ref_ts)

    tr0 = itrf.getTransformTo(inertial, t_ref)
    pv0 = tr0.transformPVCoordinates(
        PVCoordinates(Vector3D(*r0_ecef), Vector3D(*v0_ecef))
    )

    mu = Constants.WGS84_EARTH_MU
    orbit0 = CartesianOrbit(pv0, inertial, t_ref, mu)

    # Integrator builder for NumericalPropagatorBuilder
    # Keep same defaults as Case-C; can be moved into config later.
    min_step = 0.1
    max_step = 300.0
    pos_tol = 10.0
    integ_builder = DormandPrince853IntegratorBuilder(min_step, max_step, pos_tol)

    builder = NumericalPropagatorBuilder(orbit0, integ_builder, PositionAngleType.TRUE, 1000.0)
    builder.setMass(float(cfg.forces.mass_kg))

    def _find_driver_by_name(drivers, target_name: str):
        try:
            n = drivers.size()
            items = [drivers.get(i) for i in range(n)]
        except Exception:
            items = list(drivers)

        for drv in items:
            if drv.getName() == target_name:
                return drv
            
        names = [drv.getName() for drv in items]
        raise RuntimeError(
            f"ParameterDriver '{target_name}' not found. Available drivers={names}"
        )

    # Force models
    run_notes: List[str] = []

    force_bundle = build_force_model_bundle(
        itrf=itrf,
        earth=earth,
        forces=cfg.forces,
    )
    apply_force_models(builder, force_bundle)
    run_notes.extend(force_bundle.notes or [])
    run_notes.append(f"Reference inertial frame: {inertial_name}")

    def _driver0(drivers):
        try:
            return drivers.get(0)
        except Exception:
            return drivers[0]
        
    cd_driver = None
    cr_driver = None

    # Cd solve-for
    if bool(getattr(cfg.forces, "estimate_cd", False)):
        if force_bundle.drag_sensitive is None:
            run_notes.append("estimate_cd=true but drag_sensitive is None; Cd solve-for skipped.")
        else:
            cd_driver = _find_driver_by_name(
                force_bundle.drag_sensitive.getDragParametersDrivers(),
                DragSensitive.DRAG_COEFFICIENT,
            )
            cd_driver.setSelected(True)
            run_notes.append(
                f"Cd solve-for active: apriori={cd_driver.getValue()} "
                f"bounds=({cd_driver.getMinValue()}, {cd_driver.getMaxValue()})"
            )

    # Cr solve-for
    if bool(getattr(cfg.forces, "estimate_cr", False)):
        if force_bundle.radiation_sensitive is None:
            run_notes.append("estimate_cr=true but radiation_sensitive is None; Cr solve-for skipped.")
        else:
            cr_driver = _find_driver_by_name(
                force_bundle.radiation_sensitive.getRadiationParametersDrivers(),
                RadiationSensitive.REFLECTION_COEFFICIENT,
            )
            cr_driver.setSelected(True)
            run_notes.append(
                f"Cr solve-for active: apriori={cr_driver.getValue()} "
                f"bounds=({cr_driver.getMinValue()}, {cr_driver.getMaxValue()})"
            )

    # Estimator
    estimator = BatchLSEstimator(LevenbergMarquardtOptimizer(), builder)
    estimator.setMaxIterations(int(cfg.max_iterations))
    estimator.setMaxEvaluations(int(cfg.max_evaluations))

    sat = ObservableSatellite(0)

    sigma = float(cfg.od.pos_sigma_m)
    weight = 1.0

    for _, row in od_df.iterrows():
        date = to_absdate(row["dt"])
        tr = itrf.getTransformTo(inertial, date)
        p_inert = tr.transformPosition(
            Vector3D(float(row["x_m"]), float(row["y_m"]), float(row["z_m"]))
        )
        estimator.addMeasurement(Position(date, p_inert, sigma, weight, sat))

    # Run OD
    estimated = estimator.estimate()  # Java array of Propagator
    prop = estimated[0]

    # Estimated drag / SRP coefficients
    cd_est = float(cd_driver.getValue()) if cd_driver is not None else None
    cr_est = float(cr_driver.getValue()) if cr_driver is not None else None

    # State at reference epoch
    st_ref = prop.propagate(t_ref)
    pv_ref_i = st_ref.getPVCoordinates()
    ri = pv_ref_i.getPosition()
    vi = pv_ref_i.getVelocity()

    tr_ref = inertial.getTransformTo(itrf, t_ref)
    pv_ref_e = tr_ref.transformPVCoordinates(pv_ref_i)
    re = pv_ref_e.getPosition()
    ve = pv_ref_e.getVelocity()

    # OD residuals in ECEF at measurement epochs
    fit_rows: List[Dict[str, Any]] = []
    for _, row in od_df.iterrows():
        date = to_absdate(row["dt"])
        st = prop.propagate(date)
        tr = inertial.getTransformTo(itrf, date)
        p_fit = tr.transformPosition(st.getPVCoordinates().getPosition())

        p_obs = np.array([row["x_m"], row["y_m"], row["z_m"]], dtype=float)
        p_est = np.array([p_fit.getX(), p_fit.getY(), p_fit.getZ()], dtype=float)
        err = p_est - p_obs

        fit_rows.append(
            {
                "iso_utc": row["iso_utc"],
                "dx_m": float(err[0]),
                "dy_m": float(err[1]),
                "dz_m": float(err[2]),
                "err_norm_m": float(np.linalg.norm(err)),
            }
        )

    od_fit = pd.DataFrame(fit_rows)

    # Stats
    norm = od_fit["err_norm_m"].to_numpy(dtype=float)
    od_rms = float(np.sqrt(np.mean(norm**2)))
    od_p95 = float(np.percentile(norm, 95))
    od_max = float(np.max(norm))

    # Outputs
    od_solution = {
        "epoch_utc": t_ref_ts.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "frame_inertial": inertial_name,
        "frame_ecef": "ITRF",  # with IERS_2010 conventions
        "state_inertial": {
            "r_m": [float(ri.getX()), float(ri.getY()), float(ri.getZ())],
            "v_mps": [float(vi.getX()), float(vi.getY()), float(vi.getZ())],
        },
        "state_ecef": {
            "r_m": [float(re.getX()), float(re.getY()), float(re.getZ())],
            "v_mps": [float(ve.getX()), float(ve.getY()), float(ve.getZ())],
        },
        "estimated_params": {
            "cd": cd_est,
            "cr": cr_est,
        },
        "forces": force_cfg_to_dict(cfg.forces, force_bundle),
        "od_window": {
            "arc_gap_s": float(cfg.arc_gap_s),
            "selection": {
                "arc_index": cfg.od.arc_index,
                "arc_indices": cfg.od.arc_indices,
                "arc_mode": cfg.od.arc_mode,
                "time_start_utc": cfg.od.time_start_utc,
                "time_end_utc": cfg.od.time_end_utc,
            },
            "downsample_s": int(cfg.od.downsample_s),
            "pos_sigma_m": float(cfg.od.pos_sigma_m),
            "anchor": cfg.od.anchor,
        },
        "solver": {
            "type": "BatchLSEstimator(LevenbergMarquardt)",
            "max_iterations": int(cfg.max_iterations),
            "max_evaluations": int(cfg.max_evaluations),
            "notes": run_notes,
        },
    }

    # Write files
    (out_dir / "od_solution.json").write_text(
        json.dumps(od_solution, indent=2), encoding="utf-8"
    )
    od_fit.to_csv(out_dir / "od_fit.csv", index=False, encoding="utf-8-sig")
    
    od_arc_start_utc = pd.to_datetime(od_df["dt"].iloc[0], utc=True).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    od_arc_end_utc = pd.to_datetime(od_df["dt"].iloc[-1], utc=True).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    od_summary = {
        "status": "ok",
        "navsol_csv": str(cfg.navsol_csv),
        "outputs_dir": str(out_dir),

        # OD block
        "od_arc_start_utc": od_arc_start_utc,
        "od_arc_end_utc": od_arc_end_utc,
        "od_epoch_utc": t_ref_ts.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "reference_inertial_frame": inertial_name,

        "od_points": int(len(od_df)),
        "mass_kg": float(cfg.forces.mass_kg),
        "area_m2": float(cfg.forces.area_m2),
        "gravity_degree": int(cfg.forces.gravity_degree),
        "gravity_order": int(cfg.forces.gravity_order),
        "atmosphere_requested": str(cfg.forces.atmosphere),

        "cd_mode": "estimated" if cd_est is not None else "fixed",
        "cr_mode": "estimated" if cr_est is not None else "fixed",
        "apriori_cd": float(cfg.forces.cd0),
        "apriori_cr": float(cfg.forces.cr0),
        "estimated_cd": cd_est,
        "estimated_cr": cr_est,

        # current DD/Orekit metric
        "od_fit_rms_m": od_rms,
        "od_fit_p95_m": od_p95,
        "od_fit_max_m": od_max,

        # MicroCosm-style alias
        "od_fit_weighted_rms": od_rms,
        "microcosm_weighted_rms_like": od_rms,
        "od_epoch_state_rms_pos_m": None,
        "od_epoch_state_rms_vel_mps": None,

        "notes": run_notes,
        "artifacts": {
            "od_solution_json": "od_solution.json",
            "od_fit_csv": "od_fit.csv",
            "od_summary_json": "od_summary.json",
        },
    }
    
    (out_dir / "od_summary.json").write_text(
        json.dumps(od_summary, indent=2), encoding="utf-8"
    )

    return od_summary


def main(argv: Optional[List[str]] = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) < 1:
        print("Usage: python -m src.pipelines.od <config.json>")
        raise SystemExit(2)

    cfg_path = Path(argv[0])
    cfg_raw = json.loads(cfg_path.read_text(encoding="utf-8"))

    summary = run_od(cfg_raw)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
