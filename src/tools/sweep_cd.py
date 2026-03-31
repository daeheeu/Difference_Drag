from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.pipelines.od import run_od
from src.pipelines.op import run_op_loaded


CD_VALUES = [2.0, 2.2, 2.4, 2.6, 2.8]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _deepcopy_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # JSON round-trip이면 nested dict 복사에 충분
    return json.loads(json.dumps(cfg))


def _cd_tag(cd: float) -> str:
    return f"cd_{cd:.1f}".replace(".", "p")


def _apply_cd(cfg: Dict[str, Any], cd: float, root_out_dir: Path) -> Dict[str, Any]:
    cfg_i = _deepcopy_cfg(cfg)

    cfg_i.setdefault("forces", {})
    cfg_i["forces"]["cd0"] = float(cd)

    # phase-1 정책 유지: direct constant-Cd estimation 비활성
    cfg_i["forces"]["estimate_cd"] = False

    tag = _cd_tag(cd)
    cfg_i["outputs_dir"] = str(root_out_dir / tag)

    return cfg_i


def main(argv: List[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) < 1:
        print("Usage: python -m src.tools.sweep_cd <config.json>")
        raise SystemExit(2)

    cfg_path = Path(argv[0])
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)

    base_cfg = _load_json(cfg_path)

    base_out_dir = Path(base_cfg.get("outputs_dir", "outputs"))
    sweep_root = base_out_dir.parent / f"{base_out_dir.name}_cd_sweep"
    sweep_root.mkdir(parents=True, exist_ok=True)

    rows = []

    for cd in CD_VALUES:
        cfg_i = _apply_cd(base_cfg, cd, sweep_root)
        out_dir = Path(cfg_i["outputs_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[CD SWEEP] running cd0={cd:.1f} -> {out_dir}")

        # 1) OD
        od_summary = run_od(cfg_i)

        # 2) load OD solution
        od_solution_path = out_dir / "od_solution.json"
        if not od_solution_path.exists():
            raise FileNotFoundError(f"Missing OD solution: {od_solution_path}")

        od_solution = _load_json(od_solution_path)

        # 3) OP
        op_summary = run_op_loaded(od_solution, cfg_i, out_dir)

        forces_used = op_summary.get("forces", {})

        rows.append({
            "cd0": float(cd),
            "outputs_dir": str(out_dir),
            "requested_atmosphere": forces_used.get("requested_atmosphere"),
            "realized_atmosphere": forces_used.get("realized_atmosphere"),
            "od_fit_rms_m": od_summary.get("od_fit_rms_m"),
            "od_fit_p95_m": od_summary.get("od_fit_p95_m"),
            "od_fit_max_m": od_summary.get("od_fit_max_m"),
            "op_points": op_summary.get("points"),
            "op_start_utc": op_summary.get("op_start_utc"),
            "op_stop_utc": op_summary.get("op_stop_utc"),
            "cd_used_in_op": op_summary.get("cd_used"),
            "notes": " | ".join(op_summary.get("notes", [])),
        })

    df = pd.DataFrame(rows)
    summary_csv = sweep_root / "cd_sweep_summary.csv"
    df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print("\n[CD SWEEP] done")
    print(df.to_string(index=False))
    print(f"\nsummary csv: {summary_csv}")


if __name__ == "__main__":
    main()