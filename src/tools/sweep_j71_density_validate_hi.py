from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.pipelines.od import run_od
from src.pipelines.op import run_op_loaded
from src.pipelines.validate import run_validate


DENSITY_SCALES = [2.0, 2.2, 2.4, 2.6]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _deepcopy_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(cfg))


def _scale_tag(scale: float) -> str:
    return f"rho_{scale:.1f}".replace(".", "p")


def _validate_summary_filename(scale: float) -> str:
    return f"{_scale_tag(scale)}_validate_summary.json"


def _apply_density_scale(cfg: Dict[str, Any], scale: float, root_out_dir: Path) -> Dict[str, Any]:
    cfg_i = _deepcopy_cfg(cfg)

    cfg_i.setdefault("forces", {})
    cfg_i["forces"]["j71_density_scale"] = float(scale)

    tag = _scale_tag(scale)
    cfg_i["outputs_dir"] = str(root_out_dir / tag)

    return cfg_i


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


def _safe_get(d: Dict[str, Any], *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def main(argv: List[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) < 1:
        print("Usage: python -m src.tools.sweep_j71_density_validate_hi <config.json>")
        raise SystemExit(2)

    cfg_path = Path(argv[0])
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)

    base_cfg = _load_json(cfg_path)
    navsol_day2_path = _resolve_day2_navsol(base_cfg)

    base_out_dir = Path(base_cfg.get("outputs_dir", "outputs"))
    sweep_root = base_out_dir.parent / f"{base_out_dir.name}_j71_density_validate_sweep_hi"
    sweep_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    for scale in DENSITY_SCALES:
        cfg_i = _apply_density_scale(base_cfg, scale, sweep_root)
        out_dir = Path(cfg_i["outputs_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[J71 DENSITY SWEEP HI] running density_scale={scale:.1f} -> {out_dir}")

        # 1) OD
        od_summary = run_od(cfg_i)

        # 2) load OD solution
        od_solution_path = out_dir / "od_solution.json"
        if not od_solution_path.exists():
            raise FileNotFoundError(f"Missing OD solution: {od_solution_path}")
        od_solution = _load_json(od_solution_path)

        # 3) OP
        op_summary = run_op_loaded(od_solution, cfg_i, out_dir)

        # 4) Validate against day2 NavSol
        ephem_path = out_dir / "op_ephemeris.csv"
        if not ephem_path.exists():
            raise FileNotFoundError(f"Missing OP ephemeris: {ephem_path}")

        validate_summary = run_validate(
            ["validate", str(ephem_path), str(navsol_day2_path)]
        )

        # 4-1) validate_summary 파일명을 density scale 포함 형태로 저장
        default_validate_summary_path = out_dir / "validate_summary.json"
        named_validate_summary_path = out_dir / _validate_summary_filename(scale)

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

        rows.append({
            "j71_density_scale": float(scale),
            "outputs_dir": str(out_dir),
            "validate_summary_json": _validate_summary_filename(scale),

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
            "validate_radial_p95_m": _safe_get(ric, "radial", "p95_m"),
            "validate_radial_max_m": _safe_get(ric, "radial", "max_m"),

            "validate_intrack_rms_m": _safe_get(ric, "intrack", "rms_m"),
            "validate_intrack_p95_m": _safe_get(ric, "intrack", "p95_m"),
            "validate_intrack_max_m": _safe_get(ric, "intrack", "max_m"),

            "validate_crosstrack_rms_m": _safe_get(ric, "crosstrack", "rms_m"),
            "validate_crosstrack_p95_m": _safe_get(ric, "crosstrack", "p95_m"),
            "validate_crosstrack_max_m": _safe_get(ric, "crosstrack", "max_m"),

            "op_notes": " | ".join(op_summary.get("notes", [])),
        })

    df = pd.DataFrame(rows)
    summary_csv = sweep_root / "j71_density_validate_sweep_hi_summary.csv"
    df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print("\n[J71 DENSITY SWEEP HI] done")
    print(df.to_string(index=False))
    print(f"\nsummary csv: {summary_csv}")


if __name__ == "__main__":
    main()