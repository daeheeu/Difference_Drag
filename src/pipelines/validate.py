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


def _load_reference_ephem_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    need = {
        "iso_utc",
        "x_i_m", "y_i_m", "z_i_m",
        "vx_i_mps", "vy_i_mps", "vz_i_mps",
    }
    missing = need - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} does not look like reference ephemeris CSV (missing {sorted(missing)})."
        )

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
    ref_ephem_path: Optional[Path] = None


def _resolve_inputs(argv: List[str]) -> Inputs:
    # Direct file mode
    if len(argv) >= 3 and argv[1].lower().endswith(".csv"):
        ephem = Path(argv[1])
        nav = Path(argv[2])
        return Inputs(
            mode="files",
            ephem_path=ephem,
            navsol_path=nav,
            outputs_dir=ephem.parent,
        )

    # Config mode
    cfg_path = Path(argv[1])
    cfg = _load_json(cfg_path)
    out_dir = Path(cfg.get("outputs_dir", "outputs"))

    # locate source OP ephemeris
    ephem = out_dir / "op_ephemeris.csv"
    if not ephem.exists():
        legacy = out_dir / "case_c_op_ephemeris.csv"
        if legacy.exists():
            ephem = legacy

    inp = cfg.get("inputs", {}) if isinstance(cfg, dict) else {}
    validate_cfg = cfg.get("validate", {}) if isinstance(cfg, dict) else {}

    # NavSol compare input
    nav = inp.get("navsol_day2_csv") or cfg.get("navsol_day2_csv")
    nav_path = Path(nav) if nav else None

    # Reference ephemeris compare input (for day2 OD overlap compare)
    ref_ephem = (
        validate_cfg.get("reference_ephem_csv")
        or inp.get("reference_ephem_csv")
        or inp.get("day2_od_ephemeris_csv")
        or cfg.get("reference_ephem_csv")
        or cfg.get("day2_od_ephemeris_csv")
    )
    ref_ephem_path = Path(ref_ephem) if ref_ephem else None

    return Inputs(
        mode="config",
        cfg=cfg,
        outputs_dir=out_dir,
        ephem_path=ephem,
        navsol_path=nav_path,
        ref_ephem_path=ref_ephem_path,
    )


def _load_op_summary_near_ephem(ephem_path: Path) -> Tuple[Dict[str, Any], str]:
    candidates = [
        ephem_path.parent / "op_summary.json",
        ephem_path.parent / "case_c_op_summary.json",
    ]
    for p in candidates:
        if p.exists():
            return _load_json(p), p.name
    return {}, "default(EME2000)"


def _get_reference_inertial_frame(FramesFactory, IERSConventions, name: str):
    key = str(name or "EME2000").strip().upper()

    if key == "EME2000":
        return FramesFactory.getEME2000(), "EME2000"

    if key == "TOD":
        return FramesFactory.getTOD(IERSConventions.IERS_2010, True), "TOD"

    raise ValueError(f"Unsupported reference_inertial_frame for validate: {name}")


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

    validate_cfg = inp.cfg.get("validate", {}) if inp.cfg else {}
    compare_target = str(validate_cfg.get("compare_target", "day2_navsol")).strip().lower()

    if compare_target not in ("day2_navsol", "reference_ephemeris"):
        raise ValueError(
            f"Unsupported validate.compare_target: {compare_target} "
            "(expected day2_navsol or reference_ephemeris)"
        )

    if compare_target == "day2_navsol":
        if inp.navsol_path is None or not Path(inp.navsol_path).exists():
            summary = {
                "status": "ok",
                "validated": False,
                "reason": "navsol_day2_csv not provided or file not found",
                "outputs_dir": str(out_dir),
                "compare_target": compare_target,
            }
            (out_dir / "validate_summary.json").write_text(
                json.dumps(summary, indent=2),
                encoding="utf-8",
            )
            return summary

    if compare_target == "reference_ephemeris":
        if inp.ref_ephem_path is None or not Path(inp.ref_ephem_path).exists():
            summary = {
                "status": "ok",
                "validated": False,
                "reason": "reference_ephem_csv not provided or file not found",
                "outputs_dir": str(out_dir),
                "compare_target": compare_target,
            }
            (out_dir / "validate_summary.json").write_text(
                json.dumps(summary, indent=2),
                encoding="utf-8",
            )
            return summary

    ephem_path = Path(inp.ephem_path)
    if not ephem_path.exists():
        raise FileNotFoundError(f"OP ephemeris not found: {ephem_path}")

    # Load source OP ephemeris
    ephem = pd.read_csv(ephem_path)
    if "iso_utc" not in ephem.columns:
        raise ValueError(f"{ephem_path} missing iso_utc")

    ephem["t_dt"] = pd.to_datetime(ephem["iso_utc"], utc=True)
    ephem = ephem.sort_values("t_dt").reset_index(drop=True)

    cols = [
        "x_i_m", "y_i_m", "z_i_m", "vx_i_mps", "vy_i_mps", "vz_i_mps",
        "x_ecef_m", "y_ecef_m", "z_ecef_m",
    ]
    for c in cols:
        if c not in ephem.columns:
            raise ValueError(f"{ephem_path} missing column {c}")

    # Initialize orekit only once (needed for NavSol -> inertial transform path)
    orekit_data_path = inp.cfg.get("orekit_data_path") if inp.cfg else None
    extra_data_paths = inp.cfg.get("orekit_extra_data_paths") if inp.cfg else None
    init_orekit(orekit_data_path, extra_data_paths=extra_data_paths)

    from org.orekit.time import TimeScalesFactory, AbsoluteDate
    from org.orekit.frames import FramesFactory
    from org.orekit.utils import IERSConventions
    from org.hipparchus.geometry.euclidean.threed import Vector3D

    op_summary, ref_source = _load_op_summary_near_ephem(ephem_path)
    ref_name = op_summary.get("reference_inertial_frame", "EME2000")

    utc = TimeScalesFactory.getUTC()
    inertial, ref_name = _get_reference_inertial_frame(
        FramesFactory,
        IERSConventions,
        ref_name,
    )
    itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)

    t_src = ephem["t_dt"].astype("int64").to_numpy() / 1e9

    # -----------------------------
    # Build comparison dataset
    # -----------------------------
    compare_path_str = None
    compare_frame_note = None

    if compare_target == "day2_navsol":
        navsol_path = Path(inp.navsol_path)
        if not navsol_path.exists():
            raise FileNotFoundError(f"NavSol day2 not found: {navsol_path}")

        nav = _load_navsol_csv(navsol_path)

        gap_s = float(inp.cfg.get("arc_gap_s", 60.0)) if inp.cfg else 60.0
        arc_indices = None
        if inp.cfg:
            op = inp.cfg.get("op", {}) if isinstance(inp.cfg, dict) else {}
            arc_indices = op.get("arc_indices", None)
        cmp_df = _select_arcs(nav, gap_s, arc_indices)

        t_q = cmp_df["t_dt"].astype("int64").to_numpy() / 1e9
        t_min, t_max = float(t_src[0]), float(t_src[-1])
        mask = (t_q >= t_min) & (t_q <= t_max)
        cmp_df = cmp_df.loc[mask].reset_index(drop=True)

        if len(cmp_df) == 0:
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

        t_q = cmp_df["t_dt"].astype("int64").to_numpy() / 1e9
        compare_path_str = str(navsol_path)

    else:
        ref_ephem_path = Path(inp.ref_ephem_path)
        if not ref_ephem_path.exists():
            raise FileNotFoundError(f"Reference ephemeris not found: {ref_ephem_path}")

        cmp_df = _load_reference_ephem_csv(ref_ephem_path)

        cmp_summary, cmp_summary_source = _load_op_summary_near_ephem(ref_ephem_path)
        cmp_frame = cmp_summary.get("reference_inertial_frame", ref_name)
        compare_frame_note = f"{cmp_frame} via {cmp_summary_source}"

        if str(cmp_frame).strip().upper() != str(ref_name).strip().upper():
            raise ValueError(
                "Reference ephemeris frame mismatch: "
                f"source OP frame={ref_name}, reference ephem frame={cmp_frame}"
            )

        t_q = cmp_df["t_dt"].astype("int64").to_numpy() / 1e9
        t_min, t_max = float(t_src[0]), float(t_src[-1])
        mask = (t_q >= t_min) & (t_q <= t_max)
        cmp_df = cmp_df.loc[mask].reset_index(drop=True)

        if len(cmp_df) == 0:
            ephem_start = ephem["t_dt"].iloc[0].strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            ephem_stop  = ephem["t_dt"].iloc[-1].strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            ref_start = cmp_df["t_dt"].iloc[0].strftime("%Y-%m-%dT%H:%M:%S.%fZ") if len(cmp_df) > 0 else "N/A"
            ref_stop  = cmp_df["t_dt"].iloc[-1].strftime("%Y-%m-%dT%H:%M:%S.%fZ") if len(cmp_df) > 0 else "N/A"
            raise ValueError(
                "No overlapping timestamps for validation against reference ephemeris. "
                f"OP ephemeris range=[{ephem_start} ~ {ephem_stop}], "
                f"Reference ephemeris overlap range=[{ref_start} ~ {ref_stop}]"
            )

        t_q = cmp_df["t_dt"].astype("int64").to_numpy() / 1e9
        compare_path_str = str(ref_ephem_path)

    # -----------------------------
    # Interpolate source OP at comparison times
    # -----------------------------
    pred = {}
    for c in cols:
        pred[c] = _interp_series(t_src, ephem[c].to_numpy(dtype=float), t_q)

    # -----------------------------
    # Compute errors
    # -----------------------------
    out_rows = []
    norms = []
    r_list, i_list, c_list = [], [], []

    if compare_target == "day2_navsol":
        for k, row in cmp_df.iterrows():
            ts = row["t_dt"].to_pydatetime().astimezone(dt.timezone.utc)

            obs_ecef = np.array(
                [float(row["x_m"]), float(row["y_m"]), float(row["z_m"])],
                dtype=float,
            )
            pred_ecef = np.array(
                [pred["x_ecef_m"][k], pred["y_ecef_m"][k], pred["z_ecef_m"][k]],
                dtype=float,
            )
            err_ecef = obs_ecef - pred_ecef

            date = AbsoluteDate(
                ts.year, ts.month, ts.day, ts.hour, ts.minute,
                ts.second + ts.microsecond / 1e6,
                utc,
            )
            tr = itrf.getTransformTo(inertial, date)
            obs_i_v3 = tr.transformPosition(Vector3D(obs_ecef[0], obs_ecef[1], obs_ecef[2]))
            obs_i = np.array(
                [float(obs_i_v3.getX()), float(obs_i_v3.getY()), float(obs_i_v3.getZ())],
                dtype=float,
            )

            pred_i = np.array(
                [pred["x_i_m"][k], pred["y_i_m"][k], pred["z_i_m"][k]],
                dtype=float,
            )
            pred_v = np.array(
                [pred["vx_i_mps"][k], pred["vy_i_mps"][k], pred["vz_i_mps"][k]],
                dtype=float,
            )

            err_i = obs_i - pred_i
            err_norm = float(np.linalg.norm(err_i))
            norms.append(err_norm)

            r_err, i_err, c_err = _ric_components(pred_i, pred_v, err_i)
            r_list.append(r_err)
            i_list.append(i_err)
            c_list.append(c_err)

            out_rows.append({
                "iso_utc": ts.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "obs_x_ecef_m": obs_ecef[0],
                "obs_y_ecef_m": obs_ecef[1],
                "obs_z_ecef_m": obs_ecef[2],
                "pred_x_ecef_m": pred_ecef[0],
                "pred_y_ecef_m": pred_ecef[1],
                "pred_z_ecef_m": pred_ecef[2],
                "err_x_ecef_m": err_ecef[0],
                "err_y_ecef_m": err_ecef[1],
                "err_z_ecef_m": err_ecef[2],
                "err_norm_m": err_norm,
                "err_r_m": r_err,
                "err_i_m": i_err,
                "err_c_m": c_err,
            })

    else:
        has_ref_ecef = {"x_ecef_m", "y_ecef_m", "z_ecef_m"}.issubset(cmp_df.columns)

        for k, row in cmp_df.iterrows():
            ts = row["t_dt"].to_pydatetime().astimezone(dt.timezone.utc)

            ref_i = np.array(
                [float(row["x_i_m"]), float(row["y_i_m"]), float(row["z_i_m"])],
                dtype=float,
            )
            pred_i = np.array(
                [pred["x_i_m"][k], pred["y_i_m"][k], pred["z_i_m"][k]],
                dtype=float,
            )
            pred_v = np.array(
                [pred["vx_i_mps"][k], pred["vy_i_mps"][k], pred["vz_i_mps"][k]],
                dtype=float,
            )

            err_i = ref_i - pred_i
            err_norm = float(np.linalg.norm(err_i))
            norms.append(err_norm)

            r_err, i_err, c_err = _ric_components(pred_i, pred_v, err_i)
            r_list.append(r_err)
            i_list.append(i_err)
            c_list.append(c_err)

            pred_ecef = np.array(
                [pred["x_ecef_m"][k], pred["y_ecef_m"][k], pred["z_ecef_m"][k]],
                dtype=float,
            )

            if has_ref_ecef:
                ref_ecef = np.array(
                    [float(row["x_ecef_m"]), float(row["y_ecef_m"]), float(row["z_ecef_m"])],
                    dtype=float,
                )
                err_ecef = ref_ecef - pred_ecef
            else:
                ref_ecef = np.array([np.nan, np.nan, np.nan], dtype=float)
                err_ecef = np.array([np.nan, np.nan, np.nan], dtype=float)

            out_rows.append({
                "iso_utc": ts.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "ref_x_ecef_m": ref_ecef[0],
                "ref_y_ecef_m": ref_ecef[1],
                "ref_z_ecef_m": ref_ecef[2],
                "pred_x_ecef_m": pred_ecef[0],
                "pred_y_ecef_m": pred_ecef[1],
                "pred_z_ecef_m": pred_ecef[2],
                "err_x_ecef_m": err_ecef[0],
                "err_y_ecef_m": err_ecef[1],
                "err_z_ecef_m": err_ecef[2],
                "err_norm_m": err_norm,
                "err_r_m": r_err,
                "err_i_m": i_err,
                "err_c_m": c_err,
            })

    out_df = pd.DataFrame(out_rows)
    out_csv = out_dir / "op_validate.csv"
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    norms_arr = np.array(norms, dtype=float) if norms else np.array([np.nan])
    rms = float(np.sqrt(np.nanmean(norms_arr**2)))
    p95 = float(np.nanpercentile(norms_arr, 95))
    mx = float(np.nanmax(norms_arr))

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

    radial_stats = _stat(r_list)
    intrack_stats = _stat(i_list)
    crosstrack_stats = _stat(c_list)

    summary = {
        "status": "ok",
        "validated": True,
        "outputs_dir": str(out_dir),
        "op_ephemeris_csv": str(ephem_path),
        "navsol_day2_csv": str(inp.navsol_path) if compare_target == "day2_navsol" and inp.navsol_path else None,
        "reference_ephemeris_csv": str(inp.ref_ephem_path) if compare_target == "reference_ephemeris" and inp.ref_ephem_path else None,
        "reference_inertial_frame": ref_name,
        "reference_inertial_frame_source": ref_source,
        "reference_compare_frame_note": compare_frame_note,
        "compare_target": compare_target,
        "points_compared": int(len(out_df)),
        "overlap_start_utc": out_df["iso_utc"].iloc[0] if len(out_df) > 0 else None,
        "overlap_stop_utc": out_df["iso_utc"].iloc[-1] if len(out_df) > 0 else None,
        "pos_3d": {"rms_m": rms, "p95_m": p95, "max_m": mx},
        "ric": {
            "radial": radial_stats,
            "intrack": intrack_stats,
            "crosstrack": crosstrack_stats,
        },

        # flat aliases for GUI / report / MicroCosm-style comparison
        "validate_pos_3d_rms_m": rms,
        "validate_pos_3d_p95_m": p95,
        "validate_pos_3d_max_m": mx,

        "validate_ric_radial_rms_m": radial_stats["rms_m"],
        "validate_ric_radial_p95_m": radial_stats["p95_m"],
        "validate_ric_radial_max_m": radial_stats["max_m"],

        "validate_ric_intrack_rms_m": intrack_stats["rms_m"],
        "validate_ric_intrack_p95_m": intrack_stats["p95_m"],
        "validate_ric_intrack_max_m": intrack_stats["max_m"],

        "validate_ric_crosstrack_rms_m": crosstrack_stats["rms_m"],
        "validate_ric_crosstrack_p95_m": crosstrack_stats["p95_m"],
        "validate_ric_crosstrack_max_m": crosstrack_stats["max_m"],

        # headline metric for MicroCosm-like reporting
        "microcosm_headline_metric": "I_RMS",
        "microcosm_headline_value_m": intrack_stats["rms_m"],

        "artifacts": {
            "op_validate_csv": out_csv.name,
            "validate_summary_json": "validate_summary.json",
        },
    }
    (out_dir / "validate_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    summary = run_validate(sys.argv)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
