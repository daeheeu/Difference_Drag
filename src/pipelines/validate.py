from __future__ import annotations

"""Validation pipeline.

Compares OP ephemeris against day2 NavSol CSV and outputs:
- op_validate.csv
- validate_summary.json

Typical usage
    python -m src.pipelines.validate configs/case_c_od_op_navsol_2025.json

Inputs supported:
- Case-C style config (preferred)
  - outputs_dir: where op_ephemeris.csv exists
  - inputs.navsol_day2_csv OR navsol_day2_csv: comparison data
  - arc_gap_s: arc segmentation gap
  - op.arc_indices: which arcs from day2 to validate (optional)
- Direct file mode (advanced):
    python -m src.pipelines.validate outputs/op_ephemeris.csv data/navsol_day2.csv

Notes
- Uses linear interpolation on OP ephemeris to evaluate predicted PV at observation times.
- Computes both ECEF position errors and RIC errors (in inertial frame).
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import datetime as dt

import numpy as np
import pandas as pd

from src.orekit_bootstrap import init_orekit


def _parse_iso_utc(s: str) -> dt.datetime:
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return dt.datetime.fromisoformat(s).astimezone(dt.timezone.utc)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_navsol_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "iso_utc" not in df.columns:
        raise ValueError(f"{path} does not look like navsol CSV (missing iso_utc).")
    df["t_dt"] = pd.to_datetime(df["iso_utc"], utc=True)
    df = df.sort_values("t_dt").reset_index(drop=True)
    return df


def _split_arcs(df: pd.DataFrame, gap_s: float) -> List[pd.DataFrame]:
    if df.empty:
        return []
    t = df["t_dt"].to_numpy()
    arcs = []
    start = 0
    for i in range(1, len(df)):
        dt_s = (t[i] - t[i-1]) / np.timedelta64(1, "s")
        if dt_s > gap_s:
            arcs.append(df.iloc[start:i].reset_index(drop=True))
            start = i
    arcs.append(df.iloc[start:].reset_index(drop=True))
    return arcs


def _select_arcs(df: pd.DataFrame, gap_s: float, arc_indices: Optional[List[int]]) -> pd.DataFrame:
    if arc_indices is None:
        return df
    arcs = _split_arcs(df, gap_s)
    sel = []
    for idx in arc_indices:
        if 0 <= idx < len(arcs):
            sel.append(arcs[idx])
    if not sel:
        return df
    return pd.concat(sel, ignore_index=True)


@dataclass
class Inputs:
    mode: str  # "config" or "files"
    cfg: Optional[Dict[str, Any]] = None
    outputs_dir: Optional[Path] = None
    ephem_path: Optional[Path] = None
    navsol_path: Optional[Path] = None


def _resolve_inputs(argv: List[str]) -> Inputs:
    # Direct file mode
    if len(argv) >= 3 and argv[1].lower().endswith(".csv"):
        ephem = Path(argv[1])
        nav = Path(argv[2])
        return Inputs(mode="files", ephem_path=ephem, navsol_path=nav, outputs_dir=ephem.parent)

    # Config mode
    cfg_path = Path(argv[1])
    cfg = _load_json(cfg_path)
    out_dir = Path(cfg.get("outputs_dir", "outputs"))
    # locate ephemeris
    ephem = out_dir / "op_ephemeris.csv"
    if not ephem.exists():
        # legacy
        legacy = out_dir / "case_c_op_ephemeris.csv"
        if legacy.exists():
            ephem = legacy
    # locate navsol day2
    inp = cfg.get("inputs", {}) if isinstance(cfg, dict) else {}
    nav = inp.get("navsol_day2_csv") or cfg.get("navsol_day2_csv")
    nav_path = Path(nav) if nav else None
    return Inputs(mode="config", cfg=cfg, outputs_dir=out_dir, ephem_path=ephem, navsol_path=nav_path)


def _interp_series(t_src: np.ndarray, y_src: np.ndarray, t_q: np.ndarray) -> np.ndarray:
    """Linear interpolation y(t) for query times."""
    return np.interp(t_q, t_src, y_src)


def _ric_components(r: np.ndarray, v: np.ndarray, dr: np.ndarray) -> Tuple[float, float, float]:
    """Compute RIC components (Radial, In-track, Cross-track) in inertial frame."""
    r_norm = np.linalg.norm(r)
    if r_norm < 1e-9:
        return np.nan, np.nan, np.nan
    eR = r / r_norm
    h = np.cross(r, v)
    h_norm = np.linalg.norm(h)
    if h_norm < 1e-12:
        return np.nan, np.nan, np.nan
    eC = h / h_norm
    eI = np.cross(eC, eR)
    return float(np.dot(eR, dr)), float(np.dot(eI, dr)), float(np.dot(eC, dr))


def run_validate(argv: List[str]) -> Dict[str, Any]:
    if len(argv) < 2:
        raise SystemExit(2)

    inp = _resolve_inputs(argv)
    out_dir = inp.outputs_dir or Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    if inp.navsol_path is None or not Path(inp.navsol_path).exists():
        # No day2 data -> skip
        summary = {
            "status": "ok",
            "validated": False,
            "reason": "navsol_day2_csv not provided or file not found",
            "outputs_dir": str(out_dir),
        }
        (out_dir / "validate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    ephem_path = Path(inp.ephem_path)
    if not ephem_path.exists():
        raise FileNotFoundError(f"OP ephemeris not found: {ephem_path}")

    navsol_path = Path(inp.navsol_path)
    if not navsol_path.exists():
        raise FileNotFoundError(f"NavSol day2 not found: {navsol_path}")

    # Load data
    ephem = pd.read_csv(ephem_path)
    if "iso_utc" not in ephem.columns:
        raise ValueError(f"{ephem_path} missing iso_utc")

    ephem["t_dt"] = pd.to_datetime(ephem["iso_utc"], utc=True)
    ephem = ephem.sort_values("t_dt").reset_index(drop=True)

    nav = _load_navsol_csv(navsol_path)

    # Arc selection (config mode only)
    gap_s = float(inp.cfg.get("arc_gap_s", 60.0)) if inp.cfg else 60.0
    arc_indices = None
    if inp.cfg:
        op = inp.cfg.get("op", {}) if isinstance(inp.cfg, dict) else {}
        arc_indices = op.get("arc_indices", None)
    nav_sel = _select_arcs(nav, gap_s, arc_indices)

    # Initialize orekit for ECEF->inertial transform (for RIC error)
    init_orekit()
    from org.orekit.time import TimeScalesFactory, AbsoluteDate
    from org.orekit.frames import FramesFactory
    from org.orekit.utils import IERSConventions
    from org.hipparchus.geometry.euclidean.threed import Vector3D

    utc = TimeScalesFactory.getUTC()
    inertial = FramesFactory.getEME2000()
    itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)

    # Build interpolation arrays
    t_src = ephem["t_dt"].astype("int64").to_numpy() / 1e9  # seconds since epoch
    t_q = nav_sel["t_dt"].astype("int64").to_numpy() / 1e9

    # If no overlap between OP ephemeris time range and day2 NavSol times, fail fast with a clear message.
    if len(nav_sel) == 0:
        ephem_start = ephem["t_dt"].iloc[0].strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        ephem_stop  = ephem["t_dt"].iloc[-1].strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        nav_start = nav["t_dt"].iloc[0].strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        nav_stop  = nav["t_dt"].iloc[-1].strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        raise ValueError(
            "No overlapping timestamps for validation. "
            f"OP ephemeris range=[{ephem_start} ~ {ephem_stop}], "
            f"Day2 NavSol range=[{nav_start} ~ {nav_stop}]. "
            "This usually means you ran OP using an old OD/OP result (different day). "
            "Run 'Run OD' then 'Run OP' for the same day1/day2 pair, then retry Verify."
        )

    # limit to ephem range
    t_min, t_max = float(t_src[0]), float(t_src[-1])
    mask = (t_q >= t_min) & (t_q <= t_max)
    nav_sel = nav_sel.loc[mask].reset_index(drop=True)
    t_q = nav_sel["t_dt"].astype("int64").to_numpy() / 1e9

    cols = [
        "x_i_m","y_i_m","z_i_m","vx_i_mps","vy_i_mps","vz_i_mps",
        "x_ecef_m","y_ecef_m","z_ecef_m",
    ]
    for c in cols:
        if c not in ephem.columns:
            raise ValueError(f"{ephem_path} missing column {c}")

    pred = {}
    for c in cols:
        pred[c] = _interp_series(t_src, ephem[c].to_numpy(dtype=float), t_q)

    # Compute errors
    out_rows = []
    norms = []
    r_list, i_list, c_list = [], [], []
    for k, row in nav_sel.iterrows():
        ts = row["t_dt"].to_pydatetime().astimezone(dt.timezone.utc)

        obs_ecef = np.array([float(row["x_m"]), float(row["y_m"]), float(row["z_m"])], dtype=float)
        pred_ecef = np.array([pred["x_ecef_m"][k], pred["y_ecef_m"][k], pred["z_ecef_m"][k]], dtype=float)
        err_ecef = obs_ecef - pred_ecef

        # inertial obs via orekit transform
        date = AbsoluteDate(ts.year, ts.month, ts.day, ts.hour, ts.minute,
                            ts.second + ts.microsecond / 1e6, utc)
        tr = itrf.getTransformTo(inertial, date)
        obs_i_v3 = tr.transformPosition(Vector3D(obs_ecef[0], obs_ecef[1], obs_ecef[2]))
        obs_i = np.array([float(obs_i_v3.getX()), float(obs_i_v3.getY()), float(obs_i_v3.getZ())], dtype=float)

        pred_i = np.array([pred["x_i_m"][k], pred["y_i_m"][k], pred["z_i_m"][k]], dtype=float)
        pred_v = np.array([pred["vx_i_mps"][k], pred["vy_i_mps"][k], pred["vz_i_mps"][k]], dtype=float)

        err_i = obs_i - pred_i
        err_norm = float(np.linalg.norm(err_i))
        norms.append(err_norm)

        r_err, i_err, c_err = _ric_components(pred_i, pred_v, err_i)
        r_list.append(r_err); i_list.append(i_err); c_list.append(c_err)

        out_rows.append({
            "iso_utc": ts.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "obs_x_ecef_m": obs_ecef[0], "obs_y_ecef_m": obs_ecef[1], "obs_z_ecef_m": obs_ecef[2],
            "pred_x_ecef_m": pred_ecef[0], "pred_y_ecef_m": pred_ecef[1], "pred_z_ecef_m": pred_ecef[2],
            "err_x_ecef_m": err_ecef[0], "err_y_ecef_m": err_ecef[1], "err_z_ecef_m": err_ecef[2],
            "err_norm_m": err_norm,
            "err_r_m": r_err, "err_i_m": i_err, "err_c_m": c_err,
        })

    out_df = pd.DataFrame(out_rows)
    out_csv = out_dir / "op_validate.csv"
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    norms_arr = np.array(norms, dtype=float) if norms else np.array([np.nan])
    rms = float(np.sqrt(np.nanmean(norms_arr**2)))
    p95 = float(np.nanpercentile(norms_arr, 95))
    mx = float(np.nanmax(norms_arr))

    # RIC stats
    def _stat(arr):
        a = np.array(arr, dtype=float)
        if a.size == 0:
            return {"n": 0, "rms_m": float("nan"), "p95_m": float("nan"), "max_m": float("nan")}
        return {
            "n": int(a.size),
            "rms_m": float(np.sqrt(np.nanmean(a**2))),
            "p95_m": float(np.nanpercentile(np.abs(a), 95)),
            "max_m": float(np.nanmax(np.abs(a))),
        }

    summary = {
        "status": "ok",
        "validated": True,
        "outputs_dir": str(out_dir),
        "op_ephemeris_csv": str(ephem_path),
        "navsol_day2_csv": str(navsol_path),
        "points_compared": int(len(out_df)),
        "pos_3d": {"rms_m": rms, "p95_m": p95, "max_m": mx},
        "ric": {"radial": _stat(r_list), "intrack": _stat(i_list), "crosstrack": _stat(c_list)},
        "artifacts": {
            "op_validate_csv": out_csv.name,
            "validate_summary_json": "validate_summary.json",
        },
    }
    (out_dir / "validate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    summary = run_validate(sys.argv)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
