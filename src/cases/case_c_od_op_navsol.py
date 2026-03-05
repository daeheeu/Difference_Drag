from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import datetime as dt
import pandas as pd
import numpy as np

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

    time_start_utc: Optional[str] = None  # e.g. "2025-05-01T18:26:41Z"
    time_end_utc: Optional[str] = None    # e.g. "2025-05-01T18:53:35Z"
    downsample_s: int = 1
    pos_sigma_m: float = 10.0


@dataclass
class OPWindow:
    arc_indices: Optional[List[int]] = None   # day2 있을 때만 사용
    pos_sigma_m: float = 10.0                 # 검증 sigma

    # day2 없을 때 OP ephemeris 생성용(time grid)
    time_start_utc: Optional[str] = None      # 예: "2025-05-01T18:53:35Z"
    time_end_utc: Optional[str] = None        # 예: "2025-05-02T18:53:35Z"
    span_hours: float = 24.0                  # end가 없으면 start + span_hours
    step_s: int = 60                          # ephemeris 시간 간격

@dataclass
class ForceModelCfg:
    mass_kg: float = 30.0
    area_m2: float = 0.052578
    cd0: float = 2.3
    estimate_cd: bool = True

    gravity_degree: int = 20
    gravity_order: int = 20
    use_third_body: bool = True

    # Start with same MVP atmosphere style as Case A: SimpleExponentialAtmosphere
    atmosphere: str = "SIMPLE_EXP"
    rho0: float = 3.614e-13
    h0_m: float = 500000.0
    h_scale_m: float = 60000.0

# CaseCConfig에서 day2를 optional 처리
@dataclass
class CaseCConfig:
    navsol_day1_csv: str
    navsol_day2_csv: Optional[str] = None
    arc_gap_s: float = 1.5
    od: ODWindow = None
    op: OPWindow = None
    forces: ForceModelCfg = None
    outputs_dir: str = "outputs"


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
    p = Path(path)
    
    # 1) 그대로 존재하면 사용
    if p.exists() and p.suffix.lower() == ".csv":
        return load_navsol_csv(p)
    
    # 2) CSV 경로가 없으면, 같은 이름의 navSol을 찾아본다
    candidates = []
    candidates.append(p)  # original
    candidates.append(p.with_suffix(".navSol"))
    candidates.append(p.with_suffix(".navsol"))

    # 언더스코어/하이픈 혼용 대응
    candidates.append(p.with_name(p.name.replace("_", "-")))
    candidates.append(p.with_name(p.name.replace("_", "-")).with_suffix(".navSol"))
    candidates.append(p.with_name(p.name.replace("_", "-")).with_suffix(".navsol"))

    found = None
    for c in candidates:
        if c.exists():
            found = c
            break
    if found is None:
        raise FileNotFoundError(f"NavSol input not found: {p} (also tried .navSol/.navsol and '_'↔'-')")

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
            prns = toks[14:14+nsv]

            ts = dt.datetime(year, mon, day, hh, mm, sec, msec * 1000, tzinfo=dt.timezone.utc)
            iso = ts.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

            rows.append({
                "iso_utc": iso,
                "x_m": x, "y_m": y, "z_m": z,
                "valid": valid, "mode": mode,
                "nsv": nsv,
                "prns": ",".join(prns)
            })

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


def print_arc_summary(tag: str, df: pd.DataFrame, gap_s: float) -> None:
    arcs = split_arcs(df, gap_s)
    print(f"\n[{tag}] arcs={len(arcs)} (gap>{gap_s}s)")
    for i, (s, e) in enumerate(arcs):
        t0 = df.loc[s, "dt"]
        t1 = df.loc[e, "dt"]
        dur = (t1 - t0).total_seconds()
        n = e - s + 1
        print(f"  arc[{i}] {t0} -> {t1}  dur={dur:.1f}s  n={n}")


def select_od_segment(df: pd.DataFrame, gap_s: float, od: ODWindow) -> pd.DataFrame:
    """
    New supported fields in od:
      - arc_indices: List[int] (preferred over time strings)
      - arc_mode: "all" | "longest" | "longest_n" | "min_duration"
      - longest_n: int
      - min_arc_s: float
    Existing fields still supported:
      - time_start_utc/time_end_utc
      - arc_index
    """
    # time window (legacy)
    if od.time_start_utc and od.time_end_utc:
        ts = _as_dt_utc(od.time_start_utc)
        te = _as_dt_utc(od.time_end_utc)
        seg = df[(df["dt"] >= ts) & (df["dt"] <= te)].copy()
        if len(seg) < 5:
            raise ValueError("OD time window too small or no data in window.")
        return seg.reset_index(drop=True)

    arcs = split_arcs(df, gap_s)

    # explicit arc indices (recommended)
    arc_indices = getattr(od, "arc_indices", None)
    if arc_indices:
        chunks = []
        for ai in arc_indices:
            if ai < 0 or ai >= len(arcs):
                raise IndexError(f"od.arc_indices contains out-of-range arc: {ai} (arcs={len(arcs)})")
            s, e = arcs[ai]
            chunks.append(df.iloc[s:e+1])
        return pd.concat(chunks).sort_values("dt").reset_index(drop=True)

    # arc_mode auto selection
    arc_mode = getattr(od, "arc_mode", None)
    if arc_mode:
        arc_mode = arc_mode.strip().lower()

        # build arc meta
        meta = []
        for i, (s, e) in enumerate(arcs):
            t0 = df.loc[s, "dt"]
            t1 = df.loc[e, "dt"]
            dur = (t1 - t0).total_seconds()
            meta.append((i, s, e, dur))

        if arc_mode == "all":
            return df.copy().reset_index(drop=True)

        if arc_mode == "longest":
            i, s, e, _ = max(meta, key=lambda x: x[3])
            return df.iloc[s:e+1].copy().reset_index(drop=True)

        if arc_mode == "longest_n":
            n = int(getattr(od, "longest_n", 2))
            pick = sorted(meta, key=lambda x: x[3], reverse=True)[:n]
            chunks = [df.iloc[s:e+1] for (i, s, e, dur) in sorted(pick, key=lambda x: x[0])]
            return pd.concat(chunks).sort_values("dt").reset_index(drop=True)

        if arc_mode == "min_duration":
            min_arc_s = float(getattr(od, "min_arc_s", 600.0))
            pick = [x for x in meta if x[3] >= min_arc_s]
            if not pick:
                raise ValueError(f"No arcs with duration >= {min_arc_s}s")
            chunks = [df.iloc[s:e+1] for (i, s, e, dur) in pick]
            return pd.concat(chunks).sort_values("dt").reset_index(drop=True)

        raise ValueError(f"Unknown od.arc_mode: {arc_mode}")

    # single arc index (legacy)
    if od.arc_index is None:
        raise ValueError("OD selection requires arc_index, arc_indices, arc_mode, or time window.")
    if od.arc_index < 0 or od.arc_index >= len(arcs):
        raise IndexError(f"od.arc_index out of range: {od.arc_index} (arcs={len(arcs)})")
    s, e = arcs[od.arc_index]
    return df.iloc[s:e+1].copy().reset_index(drop=True)

def pick_od_reference_index(od_df: pd.DataFrame, gap_s: float, anchor: str) -> int:
    arcs = split_arcs(od_df, gap_s)
    if not arcs:
        return 0

    anchor = (anchor or "last").lower()

    if anchor == "first":
        return arcs[0][0]

    if anchor == "last":
        return arcs[-1][0]

    if anchor == "longest":
        # 가장 긴 arc의 시작 인덱스
        best = None
        for (s, e) in arcs:
            dur = (od_df.loc[e, "dt"] - od_df.loc[s, "dt"]).total_seconds()
            if best is None or dur > best[0]:
                best = (dur, s)
        return best[1]

    # fallback
    return arcs[-1][0]

def select_op_segment(df: pd.DataFrame, gap_s: float, op: OPWindow) -> pd.DataFrame:
    arcs = split_arcs(df, gap_s)
    if op.arc_indices is None:
        return df.copy().reset_index(drop=True)
    chunks = []
    for ai in op.arc_indices:
        if ai < 0 or ai >= len(arcs):
            raise IndexError(f"op.arc_indices contains out-of-range arc: {ai} (arcs={len(arcs)})")
        s, e = arcs[ai]
        chunks.append(df.iloc[s:e+1])
    return pd.concat(chunks).sort_values("dt").reset_index(drop=True)


def downsample(df: pd.DataFrame, step_s: int) -> pd.DataFrame:
    if step_s <= 1:
        return df
    t0 = df["dt"].iloc[0]
    dt_s = (df["dt"] - t0).dt.total_seconds()
    mask = (dt_s % step_s) < 1e-6
    return df.loc[mask].copy().reset_index(drop=True)

def build_op_epochs(cfg: CaseCConfig, od_df: pd.DataFrame, op_df: Optional[pd.DataFrame]) -> pd.DatetimeIndex:
    """
    OP epoch 생성 규칙
    - day2(op_df)가 있으면: op_df["dt"]를 그대로 사용
    - day2가 없으면: cfg.op.time_start_utc / time_end_utc(or span_hours) / step_s 로 time-grid 생성
    """
    if op_df is not None and len(op_df) > 0:
        return pd.DatetimeIndex(op_df["dt"])

    # day2가 없으면 time-grid 생성
    if cfg.op.time_start_utc:
        t0 = pd.to_datetime(cfg.op.time_start_utc, utc=True)
    else:
        # 기본: OD 데이터 마지막 시각
        t0 = pd.to_datetime(od_df["dt"].iloc[-1], utc=True)

    if cfg.op.time_end_utc:
        t1 = pd.to_datetime(cfg.op.time_end_utc, utc=True)
    else:
        t1 = t0 + pd.Timedelta(hours=float(cfg.op.span_hours))

    step = int(cfg.op.step_s)
    if step < 1:
        step = 1

    # pandas 기본은 start/end 포함(단, freq 격자에 정확히 맞지 않으면 end가 포함 안될 수 있음)
    return pd.date_range(start=t0, end=t1, freq=pd.Timedelta(seconds=step), tz="UTC")


def estimate_v_ecef(df: pd.DataFrame, idx0: int = 0) -> np.ndarray:
    if len(df) < idx0 + 3:
        raise ValueError("Need >=3 points from idx0 to estimate velocity.")
    r0 = df.loc[idx0, ["x_m","y_m","z_m"]].to_numpy(dtype=float)
    r2 = df.loc[idx0 + 2, ["x_m","y_m","z_m"]].to_numpy(dtype=float)
    t0 = df.loc[idx0, "dt"].value / 1e9
    t2 = df.loc[idx0 + 2, "dt"].value / 1e9
    return (r2 - r0) / (t2 - t0)


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.cases.case_c_od_op_navsol <config.json>")
        sys.exit(1)

    cfg_path = Path(sys.argv[1])
    cfg_raw = json.loads(cfg_path.read_text(encoding="utf-8"))

    cfg = CaseCConfig(
        navsol_day1_csv=cfg_raw["inputs"]["navsol_day1_csv"],
        navsol_day2_csv=cfg_raw["inputs"].get("navsol_day2_csv"),
        arc_gap_s=float(cfg_raw.get("arc_gap_s", 1.5)),
        od=ODWindow(**cfg_raw["od"]),
        op=OPWindow(**cfg_raw["op"]),
        forces=ForceModelCfg(**cfg_raw["forces"]),
        outputs_dir=str(cfg_raw.get("outputs_dir", "outputs")),
        )

    out_dir = Path(cfg.outputs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Start JVM + set classpath + load orekit-data (reusing your stable bootstrap)
    init_orekit()  # same pattern as Case A

    # Orekit imports AFTER init_orekit()
    from org.hipparchus.optim.nonlinear.vector.leastsquares import LevenbergMarquardtOptimizer
    from org.orekit.time import AbsoluteDate, TimeScalesFactory
    from org.orekit.frames import FramesFactory
    from org.orekit.utils import IERSConventions, Constants
    from org.orekit.utils import PVCoordinates
    from org.orekit.orbits import CartesianOrbit, PositionAngleType
    from org.orekit.propagation import SpacecraftState
    from org.orekit.propagation.conversion import DormandPrince853IntegratorBuilder
    from org.orekit.propagation.conversion import NumericalPropagatorBuilder
    from org.orekit.forces.gravity.potential import GravityFieldFactory
    from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel, ThirdBodyAttraction
    from org.orekit.bodies import CelestialBodyFactory, OneAxisEllipsoid
    from org.orekit.forces.drag import DragForce, IsotropicDrag
    from org.orekit.models.earth.atmosphere import SimpleExponentialAtmosphere
    from org.orekit.estimation.leastsquares import BatchLSEstimator
    from org.orekit.estimation.measurements import ObservableSatellite, Position
    from org.hipparchus.geometry.euclidean.threed import Vector3D

    # Load data
    df1 = load_navsol(cfg.navsol_day1_csv)
    
    df2 = None
    if cfg.navsol_day2_csv:
        try:
            df2 = load_navsol(cfg.navsol_day2_csv)
        except FileNotFoundError:
            df2 = None

    print_arc_summary("DAY1", df1, cfg.arc_gap_s)
    if df2 is not None:
        print_arc_summary("DAY2", df2, cfg.arc_gap_s)
    else:
        print("\n[DAY2] navsol_day2_csv not provided or file not found -> validation will be skipped.")

    od_df_full = select_od_segment(df1, cfg.arc_gap_s, cfg.od)
    od_df = downsample(od_df_full, cfg.od.downsample_s)

    op_df = None
    if df2 is not None:
        op_df = select_op_segment(df2, cfg.arc_gap_s, cfg.op)

    # Frames
    utc = TimeScalesFactory.getUTC()
    inertial = FramesFactory.getEME2000()
    itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)  # uses EOP if available in orekit-data

    earth = OneAxisEllipsoid(
        Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
        Constants.WGS84_EARTH_FLATTENING,
        itrf,
    )

    def to_absdate(ts: pd.Timestamp) -> AbsoluteDate:
        return AbsoluteDate(ts.year, ts.month, ts.day, ts.hour, ts.minute,
                            ts.second + ts.microsecond * 1e-6, utc)

    # Initial guess from OD arc first epoch
    ref_idx_full = pick_od_reference_index(od_df_full, cfg.arc_gap_s, cfg.od.anchor)

    r0_ecef = od_df_full.loc[ref_idx_full, ["x_m","y_m","z_m"]].to_numpy(dtype=float)
    v0_ecef = estimate_v_ecef(od_df_full, ref_idx_full)

    t_ref = to_absdate(od_df_full.loc[ref_idx_full, "dt"])
    tr0 = itrf.getTransformTo(inertial, t_ref)
    pv0 = tr0.transformPVCoordinates(PVCoordinates(Vector3D(*r0_ecef), Vector3D(*v0_ecef)))

    mu = Constants.WGS84_EARTH_MU
    orbit0 = CartesianOrbit(pv0, inertial, t_ref, mu)
    # state0 = SpacecraftState(orbit0, cfg.forces.mass_kg)

    # Integrator builder for NumericalPropagatorBuilder
    min_step = 0.1
    max_step = 300.0
    pos_tol = 10.0
    integ_builder = DormandPrince853IntegratorBuilder(min_step, max_step, pos_tol)

    builder = NumericalPropagatorBuilder(orbit0, integ_builder, PositionAngleType.TRUE, 1000.0)
    builder.setMass(cfg.forces.mass_kg)

    # Force models
    grav = GravityFieldFactory.getNormalizedProvider(cfg.forces.gravity_degree, cfg.forces.gravity_order)
    builder.addForceModel(HolmesFeatherstoneAttractionModel(itrf, grav))

    if cfg.forces.use_third_body:
        builder.addForceModel(ThirdBodyAttraction(CelestialBodyFactory.getSun()))
        builder.addForceModel(ThirdBodyAttraction(CelestialBodyFactory.getMoon()))

    # Drag (start with SimpleExponentialAtmosphere, same spirit as Case A :contentReference[oaicite:3]{index=3})
    if cfg.forces.atmosphere.upper() == "SIMPLE_EXP":
        atmosphere = SimpleExponentialAtmosphere(earth, cfg.forces.rho0, cfg.forces.h0_m, cfg.forces.h_scale_m)
    else:
        # fallback
        atmosphere = SimpleExponentialAtmosphere(earth, cfg.forces.rho0, cfg.forces.h0_m, cfg.forces.h_scale_m)

    drag_sensitive = IsotropicDrag(cfg.forces.area_m2, cfg.forces.cd0)
    drag_force = DragForce(atmosphere, drag_sensitive)
    builder.addForceModel(drag_force)

    if cfg.forces.estimate_cd:
        # Select Cd for estimation
        for drv in drag_sensitive.getDragParametersDrivers():
            drv.setSelected(True)

    # Estimator
    estimator = BatchLSEstimator(LevenbergMarquardtOptimizer(), builder)
    estimator.setMaxIterations(300)
    estimator.setMaxEvaluations(8000)

    sat = ObservableSatellite(0)

    # Add measurements: convert each ECEF obs to inertial and feed Position()
    sigma = float(cfg.od.pos_sigma_m)
    weight = 1.0

    for _, row in od_df.iterrows():
        date = to_absdate(row["dt"])
        tr = itrf.getTransformTo(inertial, date)
        p_inert = tr.transformPosition(Vector3D(float(row["x_m"]), float(row["y_m"]), float(row["z_m"])))
        estimator.addMeasurement(Position(date, p_inert, sigma, weight, sat))

    # Run OD
    estimated = estimator.estimate()       # returns Propagator[] (Java array)
    prop = estimated[0]               # take first fitted propagator

    # OD solution 저장 (재사용용)
    st_ref = prop.propagate(t_ref)
    pv_ref_i = st_ref.getPVCoordinates()
    ri = pv_ref_i.getPosition()
    vi = pv_ref_i.getVelocity()

    tr_ref = inertial.getTransformTo(itrf, t_ref)
    pv_ref_e = tr_ref.transformPVCoordinates(pv_ref_i)
    re = pv_ref_e.getPosition()
    ve = pv_ref_e.getVelocity()
    
    cd_est = None
    if cfg.forces.estimate_cd:
        try:
            drivers = list(drag_sensitive.getDragParametersDrivers())
            if drivers:
                cd_est = float(drivers[0].getValue())
        except Exception:
            cd_est = None

    od_solution = {
        "epoch_utc": od_df_full.loc[ref_idx_full, "dt"].strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "frame_inertial": "EME2000",
        "state_inertial": {
            "r_m": [ri.getX(), ri.getY(), ri.getZ()],
            "v_mps": [vi.getX(), vi.getY(), vi.getZ()],
        },
        "state_ecef": {
            "r_m": [re.getX(), re.getY(), re.getZ()],
            "v_mps": [ve.getX(), ve.getY(), ve.getZ()],
        },
        "estimated_params": {
            "cd": cd_est
        },
        "meta": {
            "gravity_degree": cfg.forces.gravity_degree,
            "gravity_order": cfg.forces.gravity_order,
            "use_third_body": cfg.forces.use_third_body,
            "atmosphere": cfg.forces.atmosphere,
            "mass_kg": cfg.forces.mass_kg,
            "area_m2": cfg.forces.area_m2,
        }
    }
    (out_dir / "case_c_od_solution.json").write_text(json.dumps(od_solution, indent=2), encoding="utf-8")


    # OP ephemeris (항상 생성/저장)
    op_epochs = build_op_epochs(cfg, od_df_full, op_df)

    ephem_rows = []
    for ts in op_epochs:
        date = to_absdate(ts)
        st = prop.propagate(date)
        pv_i = st.getPVCoordinates()

        # inertial PV
        ri = pv_i.getPosition()
        vi = pv_i.getVelocity()

        # ECEF PV
        tr = inertial.getTransformTo(itrf, date)
        pv_e = tr.transformPVCoordinates(pv_i)
        re = pv_e.getPosition()
        ve = pv_e.getVelocity()

        ephem_rows.append({
            "iso_utc": ts.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "x_i_m": ri.getX(), "y_i_m": ri.getY(), "z_i_m": ri.getZ(),
            "vx_i_mps": vi.getX(), "vy_i_mps": vi.getY(), "vz_i_mps": vi.getZ(),
            "x_ecef_m": re.getX(), "y_ecef_m": re.getY(), "z_ecef_m": re.getZ(),
            "vx_ecef_mps": ve.getX(), "vy_ecef_mps": ve.getY(), "vz_ecef_mps": ve.getZ(),
        })

    op_ephem = pd.DataFrame(ephem_rows)
    op_ephem.to_csv(out_dir / "case_c_op_ephemeris.csv", index=False, encoding="utf-8-sig")

    # OD fit residuals (ECEF)
    od_rows = []
    for _, row in od_df.iterrows():
        date = to_absdate(row["dt"])
        st = prop.propagate(date)
        tr = inertial.getTransformTo(itrf, date)
        p_fit = tr.transformPosition(st.getPVCoordinates().getPosition())
        p_obs = np.array([row["x_m"], row["y_m"], row["z_m"]], dtype=float)
        p_est = np.array([p_fit.getX(), p_fit.getY(), p_fit.getZ()], dtype=float)
        err = p_est - p_obs
        od_rows.append({
            "iso_utc": row["iso_utc"],
            "err_norm_m": float(np.linalg.norm(err)),
        })
    od_fit = pd.DataFrame(od_rows)
    od_fit.to_csv(out_dir / "case_c_od_fit.csv", index=False, encoding="utf-8-sig")
    od_rms = float(np.sqrt(np.mean(od_fit["err_norm_m"].to_numpy() ** 2)))

    # OP validation: propagate at all day2 epochs and compare in inertial, report 3D + RIC
    validated = False
    op_rms = None
    dR_rms = dI_rms = dC_rms = None

    if op_df is not None:
        validated = True

        op_rows = []
        for _, row in op_df.iterrows():
            date = to_absdate(row["dt"])
            st = prop.propagate(date)
            pv = st.getPVCoordinates()

            # NavSol ECEF -> inertial
            tr = itrf.getTransformTo(inertial, date)
            p_nav = tr.transformPosition(Vector3D(float(row["x_m"]), float(row["y_m"]), float(row["z_m"])))

            e = pv.getPosition().subtract(p_nav)  # OP - NavSol (inertial)
            ex, ey, ez = e.getX(), e.getY(), e.getZ()
            err_norm = math.sqrt(ex*ex + ey*ey + ez*ez)

            # RIC basis from OP state
            r = pv.getPosition()
            v = pv.getVelocity()
            r_hat = r.normalize()
            c_hat = r.crossProduct(v).normalize()
            i_hat = c_hat.crossProduct(r_hat)
            
            dR = e.dotProduct(r_hat)
            dI = e.dotProduct(i_hat)
            dC = e.dotProduct(c_hat)
            
            op_rows.append({
                "iso_utc": row["iso_utc"],
                "err_norm_m": err_norm,
                "dR_m": dR,
                "dI_m": dI,
                "dC_m": dC,
            })
            
        op_val = pd.DataFrame(op_rows)
        op_val.to_csv(out_dir / "case_c_op_validate.csv", index=False, encoding="utf-8-sig")
            
        op_rms = float(np.sqrt(np.mean(op_val["err_norm_m"].to_numpy() ** 2)))
        dR_rms = float(np.sqrt(np.mean(op_val["dR_m"].to_numpy() ** 2)))
        dI_rms = float(np.sqrt(np.mean(op_val["dI_m"].to_numpy() ** 2)))
        dC_rms = float(np.sqrt(np.mean(op_val["dC_m"].to_numpy() ** 2)))
    else:
        print("\n[VALIDATION] Skipped (no day2 navsol).")   

    summary = {
        "validated": validated,
        "od_fit_rms_m": od_rms,
        "od_solution_json": "case_c_od_solution.json",
        "op_ephemeris_csv": "case_c_op_ephemeris.csv",

        "op_rms_3d_m": op_rms,
        "op_rms_RIC_m": (None if not validated else {"R": dR_rms, "I": dI_rms, "C": dC_rms}),
        "od_points": int(len(od_df)),
        "op_points": (0 if op_df is None else int(len(op_df))),
        "od_selection": {
            "arc_index": cfg.od.arc_index,
            "time_start_utc": cfg.od.time_start_utc,
            "time_end_utc": cfg.od.time_end_utc,
            "downsample_s": cfg.od.downsample_s,
        },
        "op_selection": {
            "arc_indices": cfg.op.arc_indices,
            "time_start_utc": cfg.op.time_start_utc,
            "time_end_utc": cfg.op.time_end_utc,
            "span_hours": cfg.op.span_hours,
            "step_s": cfg.op.step_s,
        },
    }
    (out_dir / "case_c_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()