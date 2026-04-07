
"""
GUI runner for NavSol -> OD -> OP -> Validate pipelines.

Purpose:
- Avoid manual JSON config editing (common source of user error).
- Provide guided inputs, sensible defaults, and one-click execution.

Usage:
  python -m src.gui.pipeline_gui

Packaging:
- PyInstaller entry point can target this module.
"""
from __future__ import annotations

import json
import os
import sys
import queue
import threading
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

# Pipelines
from src.pipelines.od import run_od  # expects cfg dict
from src.pipelines.op import run_op as run_op_pipeline  # expects config path
from src.pipelines.validate import run_validate as run_verify_pipeline  # expects argv-style list


def _now_utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _safe_float(s: str, default: float) -> float:
    try:
        return float(s)
    except Exception:
        return default


def _safe_int(s: str, default: int) -> int:
    try:
        return int(float(s))
    except Exception:
        return default


class PipelineGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("NavSol OD/OP/Validate Pipeline")
        self.geometry("980x720")
        self.minsize(920, 650)

        self._q: "queue.Queue[tuple[str,str]]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._running = False

        self._build_ui()
        self.after(100, self._poll_log_queue)

    # -----------------------
    # UI construction
    # -----------------------
    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)

        # Top: Inputs
        inputs = ttk.LabelFrame(root, text="Inputs (필수/선택)", padding=10)
        inputs.pack(fill="x")

        self.var_day1 = tk.StringVar()
        self.var_day2 = tk.StringVar()
        self.var_outdir = tk.StringVar(value="outputs")
        self.var_cfgsave = tk.StringVar(value="")  # optional

        row = 0
        ttk.Label(inputs, text="Day1 NavSol CSV (필수)").grid(row=row, column=0, sticky="w")
        ttk.Entry(inputs, textvariable=self.var_day1, width=85).grid(row=row, column=1, sticky="we", padx=6)
        ttk.Button(inputs, text="Browse", command=self._pick_day1).grid(row=row, column=2, padx=4)
        row += 1

        ttk.Label(inputs, text="Day2 NavSol CSV (검증용, 선택)").grid(row=row, column=0, sticky="w")
        ttk.Entry(inputs, textvariable=self.var_day2, width=85).grid(row=row, column=1, sticky="we", padx=6)
        ttk.Button(inputs, text="Browse", command=self._pick_day2).grid(row=row, column=2, padx=4)
        row += 1

        ttk.Label(inputs, text="Outputs Dir (산출물 폴더)").grid(row=row, column=0, sticky="w")
        ttk.Entry(inputs, textvariable=self.var_outdir, width=85).grid(row=row, column=1, sticky="we", padx=6)
        ttk.Button(inputs, text="Browse", command=self._pick_outdir).grid(row=row, column=2, padx=4)
        row += 1

        hint = (
            "Tip) JSON config를 직접 수정하지 않아도 됩니다.\n"
            " - OP 시작 시각은 기본적으로 OD epoch(추정된 상태벡터 epoch)에서 자동 시작합니다.\n"
            " - Verify(검증)는 OP 예측(op_ephemeris.csv)과 Day2 NavSol을 비교합니다.\n"
        )
        ttk.Label(inputs, text=hint, foreground="#666").grid(row=row, column=0, columnspan=3, sticky="w", pady=(6, 0))
        inputs.columnconfigure(1, weight=1)

        # Middle: Settings (OD/OP/Forces)
        mid = ttk.Frame(root)
        mid.pack(fill="x", pady=10)

        od_box = ttk.LabelFrame(mid, text="OD Settings", padding=10)
        op_box = ttk.LabelFrame(mid, text="OP Settings", padding=10)
        fm_box = ttk.LabelFrame(mid, text="Force Model (기본값 권장)", padding=10)

        od_box.pack(side="left", fill="both", expand=True, padx=(0, 8))
        op_box.pack(side="left", fill="both", expand=True, padx=(0, 8))
        fm_box.pack(side="left", fill="both", expand=True)

        # OD settings
        self.var_arc_gap = tk.StringVar(value="60")
        self.var_arc_mode = tk.StringVar(value="longest")  # safest even if not-longest file
        self.var_anchor = tk.StringVar(value="last")
        self.var_downsample = tk.StringVar(value="10")
        self.var_pos_sigma = tk.StringVar(value="30")

        r = 0
        ttk.Label(od_box, text="Arc gap (s)").grid(row=r, column=0, sticky="w")
        ttk.Entry(od_box, textvariable=self.var_arc_gap, width=10).grid(row=r, column=1, sticky="w")
        ttk.Label(od_box, text="(NavSol 끊김 판단, longest면 60 권장)", foreground="#666").grid(row=r, column=2, sticky="w")
        r += 1

        ttk.Label(od_box, text="Arc mode").grid(row=r, column=0, sticky="w")
        ttk.Combobox(od_box, textvariable=self.var_arc_mode, width=12, values=("longest", "all", "longest_n", "min_duration"), state="readonly").grid(row=r, column=1, sticky="w")
        ttk.Label(od_box, text="(longest 권장)", foreground="#666").grid(row=r, column=2, sticky="w")
        r += 1

        ttk.Label(od_box, text="Anchor").grid(row=r, column=0, sticky="w")
        ttk.Combobox(od_box, textvariable=self.var_anchor, width=12, values=("last", "first", "longest"), state="readonly").grid(row=r, column=1, sticky="w")
        ttk.Label(od_box, text="(OD epoch 선택: last 권장)", foreground="#666").grid(row=r, column=2, sticky="w")
        r += 1

        ttk.Label(od_box, text="Downsample (s)").grid(row=r, column=0, sticky="w")
        ttk.Entry(od_box, textvariable=self.var_downsample, width=10).grid(row=r, column=1, sticky="w")
        ttk.Label(od_box, text="(예: 10 = 10초 간격)", foreground="#666").grid(row=r, column=2, sticky="w")
        r += 1

        ttk.Label(od_box, text="Pos sigma (m)").grid(row=r, column=0, sticky="w")
        ttk.Entry(od_box, textvariable=self.var_pos_sigma, width=10).grid(row=r, column=1, sticky="w")
        ttk.Label(od_box, text="(NavSol 측정 표준편차)", foreground="#666").grid(row=r, column=2, sticky="w")
        r += 1

        od_box.columnconfigure(2, weight=1)

        # OP settings
        self.var_span_h = tk.StringVar(value="30")
        self.var_step_s = tk.StringVar(value="60")
        self.var_time_start = tk.StringVar(value="")  # blank means use OD epoch

        r = 0
        ttk.Label(op_box, text="Span (hours)").grid(row=r, column=0, sticky="w")
        ttk.Entry(op_box, textvariable=self.var_span_h, width=10).grid(row=r, column=1, sticky="w")
        r += 1

        ttk.Label(op_box, text="Step (s)").grid(row=r, column=0, sticky="w")
        ttk.Entry(op_box, textvariable=self.var_step_s, width=10).grid(row=r, column=1, sticky="w")
        r += 1

        ttk.Label(op_box, text="Start UTC (optional)").grid(row=r, column=0, sticky="w")
        ttk.Entry(op_box, textvariable=self.var_time_start, width=20).grid(row=r, column=1, sticky="w")
        ttk.Label(op_box, text="(비워두면 OD epoch 사용)", foreground="#666").grid(row=r, column=2, sticky="w")
        r += 1

        fmt = "예) 2025-02-25T14:16:02Z"
        ttk.Label(op_box, text=fmt, foreground="#666").grid(row=r, column=0, columnspan=3, sticky="w")
        op_box.columnconfigure(2, weight=1)

        # Force model
        self.var_mass = tk.StringVar(value="30.0")
        self.var_area = tk.StringVar(value="0.052578")
        self.var_cd0 = tk.StringVar(value="2.3")
        self.var_est_cd = tk.BooleanVar(value=False)  # keep OFF for now
        self.var_grav_deg = tk.StringVar(value="20")
        self.var_grav_ord = tk.StringVar(value="20")
        self.var_third_body = tk.BooleanVar(value=False)
        self.var_atm = tk.StringVar(value="SIMPLE_EXP")
        self.var_rho0 = tk.StringVar(value="3.614e-13")
        self.var_h0 = tk.StringVar(value="500000")
        self.var_hs = tk.StringVar(value="60000")

        r = 0
        ttk.Label(fm_box, text="mass_kg").grid(row=r, column=0, sticky="w")
        ttk.Entry(fm_box, textvariable=self.var_mass, width=12).grid(row=r, column=1, sticky="w")
        r += 1

        ttk.Label(fm_box, text="area_m2").grid(row=r, column=0, sticky="w")
        ttk.Entry(fm_box, textvariable=self.var_area, width=12).grid(row=r, column=1, sticky="w")
        r += 1

        ttk.Label(fm_box, text="cd0").grid(row=r, column=0, sticky="w")
        ttk.Entry(fm_box, textvariable=self.var_cd0, width=12).grid(row=r, column=1, sticky="w")
        r += 1

        chk = ttk.Checkbutton(fm_box, text="estimate_cd(항력 계수 추정 ON/OFF)", variable=self.var_est_cd)
        #chk.state(["disabled"])  # enforce OFF now
        chk.grid(row=r, column=0, columnspan=2, sticky="w", pady=(0, 6))
        r += 1

        ttk.Label(fm_box, text="gravity (deg/order)").grid(row=r, column=0, sticky="w")
        rowf = ttk.Frame(fm_box)
        rowf.grid(row=r, column=1, sticky="w")
        ttk.Entry(rowf, textvariable=self.var_grav_deg, width=5).pack(side="left")
        ttk.Label(rowf, text="/").pack(side="left")
        ttk.Entry(rowf, textvariable=self.var_grav_ord, width=5).pack(side="left")
        r += 1

        ttk.Checkbutton(fm_box, text="use_third_body (Sun+Moon)", variable=self.var_third_body).grid(row=r, column=0, columnspan=2, sticky="w")
        r += 1

        ttk.Label(fm_box, text="atmosphere").grid(row=r, column=0, sticky="w")
        ttk.Combobox(fm_box, textvariable=self.var_atm, values=("SIMPLE_EXP",), width=10, state="readonly").grid(row=r, column=1, sticky="w")
        r += 1

        ttk.Label(fm_box, text="rho0").grid(row=r, column=0, sticky="w")
        ttk.Entry(fm_box, textvariable=self.var_rho0, width=12).grid(row=r, column=1, sticky="w")
        r += 1

        ttk.Label(fm_box, text="h0_m").grid(row=r, column=0, sticky="w")
        ttk.Entry(fm_box, textvariable=self.var_h0, width=12).grid(row=r, column=1, sticky="w")
        r += 1

        ttk.Label(fm_box, text="h_scale_m").grid(row=r, column=0, sticky="w")
        ttk.Entry(fm_box, textvariable=self.var_hs, width=12).grid(row=r, column=1, sticky="w")
        r += 1

        fm_box.columnconfigure(1, weight=1)

        # Bottom: Actions + log
        bottom = ttk.Frame(root)
        bottom.pack(fill="both", expand=True)

        btns = ttk.Frame(bottom)
        btns.pack(fill="x", pady=(0, 8))

        ttk.Button(btns, text="Run OD", command=self._run_od_only).pack(side="left", padx=4)
        ttk.Button(btns, text="Run OP", command=self._run_op_only).pack(side="left", padx=4)
        ttk.Button(btns, text="Run Verify", command=self._run_verify_only).pack(side="left", padx=4)
        ttk.Button(btns, text="Run OD + OP + Verify", command=self._run_all).pack(side="left", padx=4)

        ttk.Button(btns, text="Open outputs folder", command=self._open_outputs).pack(side="right", padx=4)

        self.pb = ttk.Progressbar(bottom, mode="indeterminate")
        self.pb.pack(fill="x")

        logframe = ttk.LabelFrame(bottom, text="Log", padding=8)
        logframe.pack(fill="both", expand=True)

        self.txt = tk.Text(logframe, height=18, wrap="word")
        self.txt.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(logframe, orient="vertical", command=self.txt.yview)
        sb.pack(side="right", fill="y")
        self.txt.configure(yscrollcommand=sb.set)

        self._log("Ready.")

    # -----------------------
    # Pickers
    # -----------------------
    def _pick_day1(self) -> None:
        p = filedialog.askopenfilename(
            title="Select Day1 NavSol CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if p:
            self.var_day1.set(p)

    def _pick_day2(self) -> None:
        p = filedialog.askopenfilename(
            title="Select Day2 NavSol CSV (optional)",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if p:
            self.var_day2.set(p)

    def _pick_outdir(self) -> None:
        p = filedialog.askdirectory(title="Select outputs directory")
        if p:
            self.var_outdir.set(p)

    # -----------------------
    # Logging
    # -----------------------
    def _log(self, msg: str, level: str = "INFO") -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self.txt.insert("end", f"[{ts}] {level}: {msg}\n")
        self.txt.see("end")

    def _poll_log_queue(self) -> None:
        try:
            while True:
                level, msg = self._q.get_nowait()
                self._log(msg, level)
        except queue.Empty:
            pass
        self.after(100, self._poll_log_queue)

    def _emit(self, msg: str, level: str = "INFO") -> None:
        self._q.put((level, msg))

    # -----------------------
    # Run pipeline
    # -----------------------
    def _build_cfg(self, require_day1: bool, require_day2: bool) -> Dict[str, Any]:
        day1 = self.var_day1.get().strip()
        day2 = self.var_day2.get().strip()
        outdir = self.var_outdir.get().strip() or "outputs"

        if require_day1:
            if not day1:
                raise ValueError("Day1 NavSol CSV is required for OD.")
            if not Path(day1).exists():
                raise FileNotFoundError(day1)
        else:
            if day1 and not Path(day1).exists():
                raise FileNotFoundError(day1)

        if require_day2:
            if not day2:
                raise ValueError("Day2 NavSol CSV is required for Verify.")
            if not Path(day2).exists():
                raise FileNotFoundError(day2)
        else:
            if day2 and not Path(day2).exists():
                raise FileNotFoundError(day2)

        cfg: Dict[str, Any] = {
            "inputs": {},
            "arc_gap_s": _safe_float(self.var_arc_gap.get(), 60.0),
            "od": {
                "arc_mode": self.var_arc_mode.get(),
                "anchor": self.var_anchor.get(),
                "downsample_s": _safe_int(self.var_downsample.get(), 10),
                "pos_sigma_m": _safe_float(self.var_pos_sigma.get(), 30.0),
            },
            "op": {
                "span_hours": _safe_float(self.var_span_h.get(), 30.0),
                "step_s": _safe_int(self.var_step_s.get(), 60),
            },
            "forces": {
                "mass_kg": _safe_float(self.var_mass.get(), 30.0),
                "area_m2": _safe_float(self.var_area.get(), 0.052578),
                "cd0": _safe_float(self.var_cd0.get(), 2.3),
                "estimate_cd": bool(self.var_est_cd.get()),  # enforced OFF in this phase
                "gravity_degree": _safe_int(self.var_grav_deg.get(), 20),
                "gravity_order": _safe_int(self.var_grav_ord.get(), 20),
                "use_third_body": bool(self.var_third_body.get()),
                "atmosphere": self.var_atm.get(),
                "rho0": _safe_float(self.var_rho0.get(), 3.614e-13),
                "h0_m": _safe_float(self.var_h0.get(), 500000.0),
                "h_scale_m": _safe_float(self.var_hs.get(), 60000.0),
            },
            "outputs_dir": outdir,
        }

        if day1:
            cfg["inputs"]["navsol_day1_csv"] = day1
        if day2:
            cfg["inputs"]["navsol_day2_csv"] = day2

        # optional OP start time
        ts = self.var_time_start.get().strip()
        if ts:
            cfg["op"]["time_start_utc"] = ts

        return cfg

    def _write_cfg(self, cfg: Dict[str, Any], out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = out_dir / "run_config.json"
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        return cfg_path

    def _ensure_exists(self, out_dir: Path, filename: str) -> None:
        p = out_dir / filename
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    def _run_od_only(self) -> None:
        self._start(do_od=True, do_op=False, do_verify=False)

    def _run_op_only(self) -> None:
        self._start(do_od=False, do_op=True, do_verify=False)

    def _run_verify_only(self) -> None:
        self._start(do_od=False, do_op=False, do_verify=True)

    def _run_all(self) -> None:
        self._start(do_od=True, do_op=True, do_verify=True)

    def _start(self, do_od: bool, do_op: bool, do_verify: bool) -> None:
        if self._running:
            messagebox.showwarning("Running", "A run is already in progress.")
            return

        try:
            cfg = self._build_cfg(require_day1=do_od, require_day2=do_verify)
        except Exception as e:
            messagebox.showerror("Invalid settings", str(e))
            return

        self._running = True
        self.pb.start(8)
        self._emit("Starting...", "INFO")

        def worker() -> None:
            try:
                out_dir = Path(cfg.get("outputs_dir", "outputs"))
                cfg_path = self._write_cfg(cfg, out_dir)

                self._emit(f"Config saved: {cfg_path}", "INFO")

                # --- OD ---
                if do_od:
                    self._emit("OD started...", "INFO")
                    od_summary = run_od(cfg)
                    self._emit(
                        f"OD done. RMS={od_summary.get('od_fit_rms_m'):.3f} m, points={od_summary.get('od_points')}",
                        "INFO",
                    )
                elif do_op:
                    # OD를 건너뛰고 OP만 수행하려면 기존 OD 산출물(od_solution.json)이 필요
                    self._ensure_exists(out_dir, "od_solution.json")

                # --- OP ---
                if do_op:
                    self._emit("OP started...", "INFO")
                    op_summary = run_op_pipeline(str(cfg_path))
                    op_points = op_summary.get("op_points", op_summary.get("points"))
                    op_frame = op_summary.get("reference_inertial_frame", "UNKNOWN")
                    self._emit(
                        f"OP done. points={op_points}, step_s={op_summary.get('step_s')}, frame={op_frame}",
                        "INFO",
                    )
                elif do_verify:
                    # OP를 건너뛰고 Verify만 수행하려면 기존 OP 산출물(op_ephemeris.csv)이 필요
                    if not (out_dir / "op_ephemeris.csv").exists():
                        # legacy name fallback
                        self._ensure_exists(out_dir, "case_c_op_ephemeris.csv")

                # --- Verify ---
                if do_verify:
                    self._emit("Verify started...", "INFO")
                    v_summary = run_verify_pipeline(["validate", str(cfg_path)])
                    if v_summary.get("validated"):
                        npt = v_summary.get("points_compared", None)

                        # Prefer flat aliases first
                        rms = v_summary.get("validate_pos_3d_rms_m", None)
                        p95 = v_summary.get("validate_pos_3d_p95_m", None)
                        mx  = v_summary.get("validate_pos_3d_max_m", None)

                        if rms is None:
                            pos3d = v_summary.get("pos_3d", {}) if isinstance(v_summary, dict) else {}
                            rms = pos3d.get("rms_m", None)
                            p95 = pos3d.get("p95_m", None)
                            mx  = pos3d.get("max_m", None)

                        intrack_rms = v_summary.get("validate_ric_intrack_rms_m", None)
                        if intrack_rms is None:
                            ric = v_summary.get("ric", {}) if isinstance(v_summary, dict) else {}
                            intrack_rms = ((ric.get("intrack") or {}).get("rms_m"))

                        frame = v_summary.get("reference_inertial_frame", "UNKNOWN")
                        compare_target = v_summary.get("compare_target", "UNKNOWN")

                        if rms is None:
                            self._emit(
                                "Verify done, but RMS not available in summary (check validate_summary.json).",
                                "WARN",
                            )
                        else:
                            if intrack_rms is None:
                                msg = (
                                    f"Verify done. points={npt}, frame={frame}, target={compare_target}, "
                                    f"3D RMS={rms:.3f} m, p95={p95:.3f} m, max={mx:.3f} m"
                                )
                            else:
                                msg = (
                                    f"Verify done. points={npt}, frame={frame}, target={compare_target}, "
                                    f"3D RMS={rms:.3f} m, p95={p95:.3f} m, max={mx:.3f} m, "
                                    f"I-RMS={intrack_rms:.3f} m"
                                )
                            self._emit(msg, "INFO")
                    else:
                        self._emit(f"Verify skipped/failed: {v_summary.get('reason')}", "WARN")

                self._emit("Finished.", "INFO")

            except Exception:
                tb = traceback.format_exc()
                self._emit(tb, "ERROR")
            finally:
                self._running = False
                self.pb.stop()

        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()

    def _open_outputs(self) -> None:
        outdir = self.var_outdir.get().strip() or "outputs"
        p = Path(outdir)
        if not p.exists():
            messagebox.showinfo("Info", f"Outputs folder does not exist yet: {p}")
            return
        try:
            if os.name == "nt":
                os.startfile(str(p))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                os.system(f'open "{p}"')
            else:
                os.system(f'xdg-open "{p}"')
        except Exception as e:
            messagebox.showerror("Error", str(e))


def main() -> None:
    app = PipelineGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
