from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.pipelines.od import run_od
from src.pipelines.op import run_op_loaded
from src.pipelines.validate import run_validate


DEFAULT_SCALES = [1.90, 1.95, 2.00, 2.05, 2.10]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _deepcopy_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(cfg))


def _scale_tag(scale: float) -> str:
    return f"rho_{scale:.2f}".replace(".", "p")


def _validate_summary_filename(scale: float) -> str:
    return f"{_scale_tag(scale)}_validate_summary.json"


def _resolve_day2_navsol(cfg: Dict[str, Any]) -> Path:
    inputs = cfg.get("inputs", {}) if isinstance(cfg, dict) else {}
    nav2 = inputs.get("navsol_day2_csv") or cfg.get("navsol_day2_csv")
    if not nav2:
        raise ValueError(
            "Config must contain inputs.navsol_day2_csv (or top-level navsol_day2_csv)."
        )

    p = Path(nav2)
    if not p.exists():
        raise FileNotFoundError(f"Day2 NavSol CSV not found: {p}")
    return p


def _apply_density_scale(cfg: Dict[str, Any], scale: float, root_out_dir: Path) -> Dict[str, Any]:
    cfg_i = _deepcopy_cfg(cfg)
    cfg_i.setdefault("forces", {})
    cfg_i["forces"]["j71_density_scale"] = float(scale)
    cfg_i["outputs_dir"] = str(root_out_dir / _scale_tag(scale))
    return cfg_i


def _safe_get(d: Dict[str, Any], *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _metric_value(row: Dict[str, Any], metric: str) -> float:
    if metric not in row:
        raise KeyError(f"Metric '{metric}' not found in result row. Available keys={sorted(row.keys())}")
    v = row[metric]
    if v is None:
        raise ValueError(f"Metric '{metric}' is None for row={row}")
    return float(v)


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Search best J71 density scale by OD->OP->validate outer-loop."
    )
    ap.add_argument("config", help="Case-C style config JSON with navsol_day1/day2")
    ap.add_argument(
        "--scales",
        nargs="*",
        type=float,
        default=DEFAULT_SCALES,
        help="Density scales to test (default: 1.90 1.95 2.00 2.05 2.10)",
    )
    ap.add_argument(
        "--metric",
        default="validate_pos_3d_rms_m",
        choices=[
            "validate_pos_3d_rms_m",
            "validate_intrack_rms_m",
            "validate_radial_rms_m",
        ],
        help="Metric to minimize when choosing best scale",
    )
    return ap


def main() -> None:
    ap = _build_arg_parser()
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)

    base_cfg = _load_json(cfg_path)
    navsol_day2_path = _resolve_day2_navsol(base_cfg)

    base_out_dir = Path(base_cfg.get("outputs_dir", "outputs"))
    search_root = base_out_dir.parent / f"{base_out_dir.name}_j71_density_search"
    search_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    for scale in args.scales:
        cfg_i = _apply_density_scale(base_cfg, float(scale), search_root)
        out_dir = Path(cfg_i["outputs_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[J71 SEARCH] running density_scale={scale:.2f} -> {out_dir}")

        # 1) OD
        od_summary = run_od(cfg_i)

        # 2) OP
        od_solution_path = out_dir / "od_solution.json"
        if not od_solution_path.exists():
            raise FileNotFoundError(f"Missing OD solution: {od_solution_path}")
        od_solution = _load_json(od_solution_path)

        op_summary = run_op_loaded(od_solution, cfg_i, out_dir)

        # 3) Validate
        ephem_path = out_dir / "op_ephemeris.csv"
        if not ephem_path.exists():
            raise FileNotFoundError(f"Missing OP ephemeris: {ephem_path}")

        validate_summary = run_validate(
            ["validate", str(ephem_path), str(navsol_day2_path)]
        )

        # rename validate summary for clarity
        default_validate_summary_path = out_dir / "validate_summary.json"
        named_validate_summary_path = out_dir / _validate_summary_filename(float(scale))

        validate_summary["artifacts"]["validate_summary_json"] = named_validate_summary_path.name
        named_validate_summary_path.write_text(
            json.dumps(validate_summary, indent=2),
            encoding="utf-8",
        )
        if default_validate_summary_path.exists():
            default_validate_summary_path.unlink()

        forces_used = op_summary.get("forces", {})
        pos3d = validate_summary.get("pos_3d", {})
        ric = validate_summary.get("ric", {})

        row = {
            "j71_density_scale": float(scale),
            "outputs_dir": str(out_dir),
            "validate_summary_json": named_validate_summary_path.name,

            "requested_atmosphere": forces_used.get("requested_atmosphere"),
            "realized_atmosphere": forces_used.get("realized_atmosphere"),
            "cd_used_in_op": op_summary.get("cd_used"),
            "cr_used_in_op": op_summary.get("cr_used"),

            "od_fit_rms_m": od_summary.get("od_fit_rms_m"),
            "od_fit_p95_m": od_summary.get("od_fit_p95_m"),
            "od_fit_max_m": od_summary.get("od_fit_max_m"),

            "validate_points_compared": validate_summary.get("points_compared"),
            "validate_reference_inertial_frame": validate_summary.get("reference_inertial_frame"),
            "validate_reference_inertial_frame_source": validate_summary.get("reference_inertial_frame_source"),

            "validate_pos_3d_rms_m": pos3d.get("rms_m"),
            "validate_pos_3d_p95_m": pos3d.get("p95_m"),
            "validate_pos_3d_max_m": pos3d.get("max_m"),

            "validate_radial_rms_m": _safe_get(ric, "radial", "rms_m"),
            "validate_intrack_rms_m": _safe_get(ric, "intrack", "rms_m"),
            "validate_crosstrack_rms_m": _safe_get(ric, "crosstrack", "rms_m"),

            "op_notes": " | ".join(op_summary.get("notes", [])),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    summary_csv = search_root / "j71_density_search_summary.csv"
    df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    best_row = min(rows, key=lambda r: _metric_value(r, args.metric))
    best_scale = float(best_row["j71_density_scale"])

    best_info = {
        "status": "ok",
        "metric": str(args.metric),
        "best_j71_density_scale": best_scale,
        "best_metric_value": float(best_row[args.metric]),
        "summary_csv": str(summary_csv),
        "best_outputs_dir": str(best_row["outputs_dir"]),
        "best_validate_summary_json": str(Path(best_row["outputs_dir"]) / best_row["validate_summary_json"]),
    }
    _write_json(search_root / "best_j71_density_scale.json", best_info)

    # also emit best-tuned config
    tuned_cfg = _deepcopy_cfg(base_cfg)
    tuned_cfg.setdefault("forces", {})
    tuned_cfg["forces"]["j71_density_scale"] = best_scale
    tuned_cfg["outputs_dir"] = str(base_out_dir.parent / f"{base_out_dir.name}_rho_best")

    tuned_cfg_path = search_root / "best_j71_density_scale_config.json"
    _write_json(tuned_cfg_path, tuned_cfg)

    print("\n[J71 SEARCH] done")
    print(df.to_string(index=False))
    print()
    print(json.dumps(best_info, indent=2))
    print(f"\nsummary csv: {summary_csv}")
    print(f"best info : {search_root / 'best_j71_density_scale.json'}")
    print(f"best cfg  : {tuned_cfg_path}")


if __name__ == "__main__":
    main()