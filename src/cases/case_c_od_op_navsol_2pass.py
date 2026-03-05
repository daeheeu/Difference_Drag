# src/cases/case_c_od_op_navsol_2pass.py
from __future__ import annotations

import json
import math
import sys
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd

from src.orekit_bootstrap import init_orekit


# -----------------------------
# Config
# -----------------------------
@dataclass
class ODWindow:
    arc_index: Optional[int] = None
    arc_indices: Optional[List[int]] = None
    arc_mode: Optional[str] = None
    longest_n: int = 2
    min_arc_s: float = 600.0
    anchor: str = "last"
    time_start_utc: Optional[str] = None
    time_end_utc: Optional[str] = None


@dataclass
class OPWindow:
    arc_indices: Optional[List[int]] = None
    pos_sigma_m: float = 10.0


@dataclass
class ForceModelCfg:
    # spacecraft
    mass_kg: float = 30.0
    area_m2: float = 0.052578

    # drag parameter
    cd0: float = 2.3
    estimate_cd: bool = True
    cd_min: float = 1.0
    cd_max: float = 5.0
    cd_scale: float = 0.1

    # gravity / 3rd body
    gravity_degree: int = 20
    gravity_order: int = 20
    use_third_body: bool = True

    # drag model (MVP)
    use_drag: bool = True
    atmosphere: str = "SIMPLE_EXP"
    rho0: float = 3.614e-13
    h0_m: float = 500000.0
    h_scale_m: float = 60000.0


@dataclass
class ODPassCfg:
    name: str = "pass"
    downsample_s: int = 10
    pos_sigma_m: float = 30.0

    # optional per-pass overrides
    estimate_cd: Optional[bool] = None
    gravity_degree: Optional[int] = None
    gravity_order: Optional[int] = None
    use_third_body: Optional[bool] = None
    use_drag: Optional[bool] = None

    # optimizer limits
    max_iterations: int = 80
    max_evaluations: int = 200


@dataclass
class CaseC2Config:
    navsol_day1_csv: str
    navsol_day2_csv: str
    arc_gap_s: float
    od: ODWindow
    op: OPWindow
    forces: ForceModelCfg
    od_passes: List[ODPassCfg]
    outputs_dir: str
    inertial_frame: str = "EME2000"  # "GCRF" recommended if matching MicroCosm-like GCRS


# -----------------------------
# NavSol loading
# -----------------------------
def load_navsol_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"iso_utc", "x_m", "y_m", "z_m"}
    if not need.issubset(df.columns):
        raise ValueError(f"NavSol CSV must contain columns: {sorted(need)}")
    df["dt"] = pd.to_datetime(df["iso_utc"], utc=True)
    df = df.sort_values("dt").reset_index(drop=True)
    return df


def load_navsol(path: str | Path) -> pd.DataFrame:
    p = Path(path)

    # 1) as-is csv
    if p.exists() and p.suffix.lower() == ".csv":
        return load_navsol_csv(p)

    # 2) try sibling .navSol/.navsol and '_'↔'-'
    candidates = [
        p,
        p.with_suffix(".navSol"),
        p.with_suffix(".navsol"),
        p.with_name(p.name.replace("_", "-")),
        p.with_name(p.name.replace("_", "-")).with_suffix(".navSol"),
        p.with_name(p.name.replace("_", "-")).with_suffix(".navsol"),
    ]
    found = next((c for c in candidates if c.exists()), None)
    if found is None:
        raise FileNotFoundError(
            f"NavSol input not found: {p} (also tried .navSol/.navsol and '_'↔'-')"
        )

    if found.suffix.lower() == ".csv":
        return load_navsol_csv(found)

    # ---- parse .navSol text ----
    rows = []
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


# -----------------------------
# Arc helpers
# -----------------------------
def split_arcs(df: pd.DataFrame, gap_s: float) -> List[Tuple[int, int]]:
    if len(df) == 0:
        return []
    arcs = []
    s = 0
    for i in range(1, len(df)):
        dt_s = (df.loc[i, "dt"] - df.loc[i - 1, "dt"]).total_seconds()
        if dt_s > gap_s:
            arcs.append((s, i - 1))
            s = i
    arcs.append((s, len(df) - 1))
    return arcs


def print_arc_summary(tag: str, df: pd.DataFrame, gap_s: float) -> None:
    arcs = split_arcs(df, gap_s)
    print(f"\n[{tag}] points={len(df)}, arcs={len(arcs)}, gap_s={gap_s}")
    for i, (s, e) in enumerate(arcs):
        t0 = df.loc[s, "dt"]
        t1 = df.loc[e, "dt"]
        dur = (t1 - t0).total_seconds()
        print(f"  arc[{i:02d}] idx=({s}-{e}) dur={dur:.1f}s  {t0} ~ {t1}")


def _as_dt_utc(s: str) -> pd.Timestamp:
    return pd.to_datetime(s, utc=True)


def select_od_segment(df: pd.DataFrame, gap_s: float, od: ODWindow) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)

    # explicit time window
    if od.time_start_utc and od.time_end_utc:
        t0 = _as_dt_utc(od.time_start_utc)
        t1 = _as_dt_utc(od.time_end_utc)
        cut = df[(df["dt"] >= t0) & (df["dt"] <= t1)].copy()
        if len(cut) == 0:
            raise ValueError("OD time window selection produced empty set.")
        return cut.reset_index(drop=True)

    arcs = split_arcs(df, gap_s)

    # explicit single arc
    if od.arc_index is not None:
        ai = int(od.arc_index)
        if ai < 0 or ai >= len(arcs):
            raise IndexError(f"od.arc_index out of range: {ai} (arcs={len(arcs)})")
        s, e = arcs[ai]
        return df.iloc[s : e + 1].copy().reset_index(drop=True)

    # explicit list
    if od.arc_indices:
        chunks = []
        for ai in od.arc_indices:
            if ai < 0 or ai >= len(arcs):
                raise IndexError(f"od.arc_indices out of range: {ai} (arcs={len(arcs)})")
            s, e = arcs[ai]
            chunks.append(df.iloc[s : e + 1])
        return pd.concat(chunks).sort_values("dt").reset_index(drop=True)

    # arc_mode
    if od.arc_mode:
        mode = od.arc_mode.lower()
        meta = []
        for i, (s, e) in enumerate(arcs):
            dur = (df.loc[e, "dt"] - df.loc[s, "dt"]).total_seconds()
            meta.append((i, s, e, dur))

        if mode == "all":
            return df.copy().reset_index(drop=True)

        if mode == "longest":
            i, s, e, _ = max(meta, key=lambda x: x[3])
            return df.iloc[s : e + 1].copy().reset_index(drop=True)

        if mode == "longest_n":
            n = int(od.longest_n)
            pick = sorted(meta, key=lambda x: x[3], reverse=True)[:n]
            chunks = [df.iloc[s : e + 1] for (_, s, e, _) in pick]
            return pd.concat(chunks).sort_values("dt").reset_index(drop=True)

        raise ValueError(f"Unknown od.arc_mode: {od.arc_mode}")

    # default: all
    return df.copy().reset_index(drop=True)


def select_op_segment(df: pd.DataFrame, gap_s: float, op: OPWindow) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    arcs = split_arcs(df, gap_s)
    if op.arc_indices is None:
        return df
    chunks = []
    for ai in op.arc_indices:
        if ai < 0 or ai >= len(arcs):
            raise IndexError(f"op.arc_indices out of range: {ai} (arcs={len(arcs)})")
        s, e = arcs[ai]
        chunks.append(df.iloc[s : e + 1])
    return pd.concat(chunks).sort_values("dt").reset_index(drop=True)


def downsample(df: pd.DataFrame, step_s: int) -> pd.DataFrame:
    if step_s <= 1:
        return df
    t0 = df["dt"].iloc[0]
    dt_s = (df["dt"] - t0).dt.total_seconds()
    mask = (dt_s % step_s) < 1e-6
    return df.loc[mask].copy().reset_index(drop=True)


def pick_od_reference_index(od_df: pd.DataFrame, gap_s: float, anchor: str) -> int:
    arcs = split_arcs(od_df, gap_s)
    if not arcs:
        return 0
    a = (anchor or "last").lower()
    if a == "first":
        return arcs[0][0]
    if a == "last":
        return arcs[-1][0]
    if a == "longest":
        best = None
        for (s, e) in arcs:
            dur = (od_df.loc[e, "dt"] - od_df.loc[s, "dt"]).total_seconds()
            if best is None or dur > best[0]:
                best = (dur, s)
        return best[1]
    return arcs[-1][0]


def estimate_v_ecef(df: pd.DataFrame, idx0: int = 0) -> np.ndarray:
    if len(df) < idx0 + 3:
        raise ValueError("Need >=3 points from idx0 to estimate velocity.")
    r0 = df.loc[idx0, ["x_m", "y_m", "z_m"]].to_numpy(dtype=float)
    r2 = df.loc[idx0 + 2, ["x_m", "y_m", "z_m"]].to_numpy(dtype=float)
    t0 = df.loc[idx0, "dt"].value / 1e9
    t2 = df.loc[idx0 + 2, "dt"].value / 1e9
    return (r2 - r0) / (t2 - t0)


def _merge_forces(base: ForceModelCfg, p: ODPassCfg) -> ForceModelCfg:
    f = ForceModelCfg(**vars(base))
    if p.estimate_cd is not None:
        f.estimate_cd = bool(p.estimate_cd)
    if p.gravity_degree is not None:
        f.gravity_degree = int(p.gravity_degree)
    if p.gravity_order is not None:
        f.gravity_order = int(p.gravity_order)
    if p.use_third_body is not None:
        f.use_third_body = bool(p.use_third_body)
    if p.use_drag is not None:
        f.use_drag = bool(p.use_drag)
    return f


# -----------------------------
# Main
# -----------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.cases.case_c_od_op_navsol_2pass <config.json>")
        sys.exit(1)

    cfg_path = Path(sys.argv[1])
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))

    cfg = CaseC2Config(
        navsol_day1_csv=raw["inputs"]["navsol_day1_csv"],
        navsol_day2_csv=raw["inputs"]["navsol_day2_csv"],
        arc_gap_s=float(raw.get("arc_gap_s", 1.5)),
        od=ODWindow(**raw["od"]),
        op=OPWindow(**raw["op"]),
        forces=ForceModelCfg(**raw["forces"]),
        od_passes=[ODPassCfg(**p) for p in raw.get("od_passes", [])],
        outputs_dir=str(raw.get("outputs_dir", "outputs")),
        inertial_frame=str(raw.get("inertial_frame", "EME2000")),
    )

    if not cfg.od_passes:
        raise ValueError("config must include non-empty 'od_passes' for 2-pass pipeline")

    out_dir = Path(cfg.outputs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Start JVM + load orekit-data
    init_orekit()

    # Orekit imports AFTER init_orekit()
    from org.hipparchus.optim.nonlinear.vector.leastsquares import LevenbergMarquardtOptimizer
    from org.hipparchus.geometry.euclidean.threed import Vector3D

    from org.orekit.time import AbsoluteDate, TimeScalesFactory
    from org.orekit.frames import FramesFactory
    from org.orekit.utils import IERSConventions, Constants
    from org.orekit.utils import PVCoordinates

    from org.orekit.orbits import CartesianOrbit, PositionAngleType
    from org.orekit.propagation.conversion import DormandPrince853IntegratorBuilder, NumericalPropagatorBuilder

    from org.orekit.forces.gravity.potential import GravityFieldFactory
    from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel, ThirdBodyAttraction
    from org.orekit.bodies import CelestialBodyFactory, OneAxisEllipsoid
    from org.orekit.forces.drag import DragForce, IsotropicDrag
    from org.orekit.models.earth.atmosphere import SimpleExponentialAtmosphere

    from org.orekit.estimation.leastsquares import BatchLSEstimator
    from org.orekit.estimation.measurements import ObservableSatellite, Position

    # Load data
    df1 = load_navsol(cfg.navsol_day1_csv)
    df2 = load_navsol(cfg.navsol_day2_csv)

    print_arc_summary("DAY1", df1, cfg.arc_gap_s)
    print_arc_summary("DAY2", df2, cfg.arc_gap_s)

    od_df_full = select_od_segment(df1, cfg.arc_gap_s, cfg.od)
    op_df = select_op_segment(df2, cfg.arc_gap_s, cfg.op)

    # Frames / time
    utc = TimeScalesFactory.getUTC()

    inertial_name = cfg.inertial_frame.strip().upper()
    if inertial_name in ("GCRF", "GCRS"):
        inertial = FramesFactory.getGCRF()
    elif inertial_name in ("EME2000", "J2000"):
        inertial = FramesFactory.getEME2000()
    else:
        raise ValueError(f"Unsupported inertial_frame: {cfg.inertial_frame} (use GCRF or EME2000)")

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

    # Initial guess epoch picked from FULL OD (not downsampled)
    ref_idx = pick_od_reference_index(od_df_full, cfg.arc_gap_s, cfg.od.anchor)

    r0_ecef = od_df_full.loc[ref_idx, ["x_m", "y_m", "z_m"]].to_numpy(dtype=float)
    v0_ecef = estimate_v_ecef(od_df_full, ref_idx)
    t_ref = to_absdate(od_df_full.loc[ref_idx, "dt"])

    tr0 = itrf.getTransformTo(inertial, t_ref)
    pv0 = tr0.transformPVCoordinates(PVCoordinates(Vector3D(*r0_ecef), Vector3D(*v0_ecef)))

    mu = Constants.WGS84_EARTH_MU
    orbit_init = CartesianOrbit(pv0, inertial, t_ref, mu)

    # Integrator builder
    min_step = 0.1
    max_step = 300.0
    pos_tol = 10.0
    integ_builder = DormandPrince853IntegratorBuilder(min_step, max_step, pos_tol)

    def make_atmosphere(forces: ForceModelCfg):
        # 지금은 SIMPLE_EXP만. (Jacchia71은 Orekit에 기본 내장 모델이 아니라서 별도 구현/연동 필요)
        if forces.atmosphere.upper() == "SIMPLE_EXP":
            return SimpleExponentialAtmosphere(earth, forces.rho0, forces.h0_m, forces.h_scale_m)
        raise ValueError(f"Unsupported atmosphere: {forces.atmosphere} (currently only SIMPLE_EXP)")

    def make_builder(orbit0: CartesianOrbit, forces: ForceModelCfg):
        builder = NumericalPropagatorBuilder(orbit0, integ_builder, PositionAngleType.TRUE, 1000.0)
        builder.setMass(forces.mass_kg)

        grav = GravityFieldFactory.getNormalizedProvider(forces.gravity_degree, forces.gravity_order)
        builder.addForceModel(HolmesFeatherstoneAttractionModel(itrf, grav))

        if forces.use_third_body:
            builder.addForceModel(ThirdBodyAttraction(CelestialBodyFactory.getSun()))
            builder.addForceModel(ThirdBodyAttraction(CelestialBodyFactory.getMoon()))

        drag_sensitive = None
        if forces.use_drag:
            atmosphere = make_atmosphere(forces)
            drag_sensitive = IsotropicDrag(forces.area_m2, forces.cd0)
            builder.addForceModel(DragForce(atmosphere, drag_sensitive))

        return builder, drag_sensitive

    def configure_cd_driver(drag_sensitive, forces: ForceModelCfg):
        if drag_sensitive is None:
            return None
        drivers = list(drag_sensitive.getDragParametersDrivers())
        if not drivers:
            return None
        # typically only Cd
        drv = drivers[0]
        try:
            drv.setMinValue(float(forces.cd_min))
            drv.setMaxValue(float(forces.cd_max))
            drv.setScale(float(forces.cd_scale))
            drv.setReferenceValue(float(forces.cd0))
        except Exception:
            # some orekit versions may not expose all setters via JPype the same way;
            # still proceed with selection at least
            pass

        drv.setSelected(bool(forces.estimate_cd))
        return drv

    def compute_od_fit_csv(prop, name: str) -> float:
        rows = []
        for _, row in od_df_full.iterrows():
            date = to_absdate(row["dt"])
            st = prop.propagate(date)
            tr = inertial.getTransformTo(itrf, date)
            p_fit = tr.transformPosition(st.getPVCoordinates().getPosition())
            p_obs = np.array([row["x_m"], row["y_m"], row["z_m"]], dtype=float)
            p_est = np.array([p_fit.getX(), p_fit.getY(), p_fit.getZ()], dtype=float)
            err = p_est - p_obs
            rows.append({"iso_utc": row["iso_utc"], "err_norm_m": float(np.linalg.norm(err))})
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / f"case_c2_od_fit_{name}.csv", index=False, encoding="utf-8-sig")
        return float(np.sqrt(np.mean(df["err_norm_m"].to_numpy() ** 2)))

    # -----------------------------
    # 2-pass OD loop
    # -----------------------------
    pass_summaries: List[Dict[str, Any]] = []
    prop_final = None

    for p in cfg.od_passes:
        forces_p = _merge_forces(cfg.forces, p)
        od_df_used = downsample(od_df_full, p.downsample_s)

        builder, drag_sensitive = make_builder(orbit_init, forces_p)
        cd_drv = configure_cd_driver(drag_sensitive, forces_p)

        estimator = BatchLSEstimator(LevenbergMarquardtOptimizer(), builder)
        estimator.setMaxIterations(int(p.max_iterations))
        estimator.setMaxEvaluations(int(p.max_evaluations))

        sat = ObservableSatellite(0)
        sigma = float(p.pos_sigma_m)
        weight = 1.0

        for _, row in od_df_used.iterrows():
            date = to_absdate(row["dt"])
            tr = itrf.getTransformTo(inertial, date)
            p_inert = tr.transformPosition(Vector3D(float(row["x_m"]), float(row["y_m"]), float(row["z_m"])))
            estimator.addMeasurement(Position(date, p_inert, sigma, weight, sat))

        estimated = estimator.estimate()
        prop = estimated[0]
        prop_final = prop

        od_rms = compute_od_fit_csv(prop, p.name)

        cd_est = None
        if cd_drv is not None and forces_p.estimate_cd:
            try:
                cd_est = float(cd_drv.getValue())
            except Exception:
                cd_est = None

        # next pass initial orbit = current solution at t_ref
        orbit_init = prop.propagate(t_ref).getOrbit()

        info = {
            "pass": p.name,
            "downsample_s": p.downsample_s,
            "pos_sigma_m": p.pos_sigma_m,
            "max_iterations": p.max_iterations,
            "max_evaluations": p.max_evaluations,
            "forces": {
                "estimate_cd": forces_p.estimate_cd,
                "cd0": forces_p.cd0,
                "cd_min": forces_p.cd_min,
                "cd_max": forces_p.cd_max,
                "cd_scale": forces_p.cd_scale,
                "cd_est": cd_est,
                "gravity_degree": forces_p.gravity_degree,
                "gravity_order": forces_p.gravity_order,
                "use_third_body": forces_p.use_third_body,
                "use_drag": forces_p.use_drag,
                "atmosphere": forces_p.atmosphere,
            },
            "od_points_used": int(len(od_df_used)),
            "od_fit_rms_m_full": od_rms,
        }
        pass_summaries.append(info)

        print("\n--- OD PASS DONE ---")
        print(json.dumps(info, indent=2))

    if prop_final is None:
        raise RuntimeError("OD failed: no propagator produced")

    # -----------------------------
    # OP validation (final propagator)
    # -----------------------------
    op_rows = []
    for _, row in op_df.iterrows():
        date = to_absdate(row["dt"])
        st = prop_final.propagate(date)
        pv = st.getPVCoordinates()

        # NavSol ECEF -> inertial
        tr = itrf.getTransformTo(inertial, date)
        p_nav = tr.transformPosition(Vector3D(float(row["x_m"]), float(row["y_m"]), float(row["z_m"])))

        e = pv.getPosition().subtract(p_nav)  # OP - NavSol (inertial)
        ex, ey, ez = e.getX(), e.getY(), e.getZ()
        err_norm = math.sqrt(ex * ex + ey * ey + ez * ez)

        # RIC basis from OP state
        r = pv.getPosition()
        v = pv.getVelocity()
        r_hat = r.normalize()
        c_hat = r.crossProduct(v).normalize()
        i_hat = c_hat.crossProduct(r_hat)

        dR = e.dotProduct(r_hat)
        dI = e.dotProduct(i_hat)
        dC = e.dotProduct(c_hat)

        op_rows.append(
            {
                "iso_utc": row["iso_utc"],
                "err_norm_m": err_norm,
                "dR_m": dR,
                "dI_m": dI,
                "dC_m": dC,
            }
        )

    op_val = pd.DataFrame(op_rows)
    op_val.to_csv(out_dir / "case_c2_op_validate.csv", index=False, encoding="utf-8-sig")

    op_rms = float(np.sqrt(np.mean(op_val["err_norm_m"].to_numpy() ** 2)))
    dR_rms = float(np.sqrt(np.mean(op_val["dR_m"].to_numpy() ** 2)))
    dI_rms = float(np.sqrt(np.mean(op_val["dI_m"].to_numpy() ** 2)))
    dC_rms = float(np.sqrt(np.mean(op_val["dC_m"].to_numpy() ** 2)))

    summary = {
        "inertial_frame": cfg.inertial_frame,
        "od_passes": pass_summaries,
        "op_rms_3d_m": op_rms,
        "op_rms_RIC_m": {"R": dR_rms, "I": dI_rms, "C": dC_rms},
        "od_points_full": int(len(od_df_full)),
        "op_points": int(len(op_df)),
        "od_selection": {
            "arc_index": cfg.od.arc_index,
            "arc_indices": cfg.od.arc_indices,
            "arc_mode": cfg.od.arc_mode,
            "time_start_utc": cfg.od.time_start_utc,
            "time_end_utc": cfg.od.time_end_utc,
            "anchor": cfg.od.anchor,
        },
        "op_selection": {"arc_indices": cfg.op.arc_indices},
    }

    (out_dir / "case_c2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== FINAL SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()