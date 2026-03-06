"""
Extract Navigation Solution (NavSol) records from raw GNSS receiver log files.

Supports:
- NovAtel ASCII BESTXYZ (e.g., #BESTXYZA, #BESTXYZ)
- Optional NMEA GPGSA to attach PRN list (satellites used)
- Outputs:
  1) navSol text file
  2) CSV (for QA/plotting)

Typical usage:
  python -m navsol_extract --in "20250502_*.dat" --out outputs/navsol_2025-05-02.navSol --csv outputs/navsol_2025-05-02.csv

Notes:
- GPS->UTC conversion needs leap seconds. Default 18 (good for 2017-2025). Override if needed.
- “Complete record” filtering: only lines that end with "*XXXXXXXX" (8-hex CRC) and parse correctly are kept.
"""

import argparse
import csv
import datetime as dt
import glob
import os
import re
import sys
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

GPS_EPOCH = dt.datetime(1980, 1, 6, tzinfo=dt.timezone.utc)

BESTXYZ_RE = re.compile(r'^#BESTXYZ[A-Z]?,.*\*[0-9A-Fa-f]{8}\s*$')
GPGSA_RE = re.compile(r'^\$GPGSA,.*\*[0-9A-Fa-f]{2}\s*$')


def nmea_checksum_ok(sentence: str) -> bool:
    s = sentence.strip()
    if not s.startswith('$') or '*' not in s:
        return False
    data, chk = s[1:].split('*', 1)
    try:
        chk_val = int(chk[:2], 16)
    except Exception:
        return False
    c = 0
    for ch in data:
        c ^= ord(ch)
    return c == chk_val


def gps_to_utc(week: int, sow: float, leap_seconds: int) -> dt.datetime:
    gps_dt = GPS_EPOCH + dt.timedelta(weeks=week, seconds=sow)
    return gps_dt - dt.timedelta(seconds=leap_seconds)


def secms5(t: dt.datetime) -> str:
    # SSmmm as 5 digits (seconds*1000 + milliseconds)
    return f"{t.second * 1000 + (t.microsecond // 1000):05d}"


def parse_bestxyz_line(line: str) -> Optional[Tuple[int, float, float, float, float, str, str]]:
    """
    Returns:
      gps_week, gps_sow, x, y, z, solstat, postype
    """
    s = line.strip()
    try:
        before_crc, _crc = s.rsplit('*', 1)
        header, data = before_crc.split(';', 1)
        h = header.split(',')
        # NovAtel BESTXYZ header fields: ... ,GPSWeek,GPSSec,...
        week = int(h[5])
        sow = float(h[6])
        d = data.split(',')
        solstat, postype = d[0], d[1]
        x, y, z = float(d[2]), float(d[3]), float(d[4])
        return week, sow, x, y, z, solstat, postype
    except Exception:
        return None


def extract_records(paths: List[str], leap_seconds: int, valid_only: bool) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    last_prns: List[int] = []
    have_prns = False

    for path in paths:
        with open(path, "r", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue

                # PRNs from GPGSA (optional)
                if line.startswith("$GPGSA"):
                    if GPGSA_RE.match(line) and nmea_checksum_ok(line):
                        parts = line.split('*')[0].split(',')
                        prns = []
                        for sv in parts[3:15]:
                            if sv:
                                try:
                                    prns.append(int(sv))
                                except Exception:
                                    pass
                        last_prns = prns
                        have_prns = True
                    continue

                # BESTXYZ solution
                if line.startswith("#BESTXYZ"):
                    if not BESTXYZ_RE.match(line):
                        continue  # incomplete/truncated line
                    parsed = parse_bestxyz_line(line)
                    if parsed is None:
                        continue

                    week, sow, x, y, z, solstat, postype = parsed
                    valid = 1 if solstat == "SOL_COMPUTED" else 0
                    if valid_only and valid != 1:
                        continue

                    utc = gps_to_utc(week, sow, leap_seconds)
                    prns = last_prns if have_prns else []

                    rows.append({
                        "dt_utc": utc,
                        "source": os.path.basename(path),
                        "gps_week": week,
                        "gps_sow": sow,
                        "solstat": solstat,
                        "postype": postype,
                        "valid": valid,
                        "mode": 2,
                        "x_m": x,
                        "y_m": y,
                        "z_m": z,
                        "nsv": len(prns),
                        "prns": prns
                    })

    rows.sort(key=lambda r: r["dt_utc"])
    return rows


def split_arcs(rows: List[Dict[str, Any]], max_gap_s: float) -> List[List[Dict[str, Any]]]:
    if not rows:
        return []
    arcs: List[List[Dict[str, Any]]] = []
    cur = [rows[0]]
    for r in rows[1:]:
        dt_prev = cur[-1]["dt_utc"]
        dt_now = r["dt_utc"]
        gap = (dt_now - dt_prev).total_seconds()
        if gap > max_gap_s:
            arcs.append(cur)
            cur = [r]
        else:
            cur.append(r)
    arcs.append(cur)
    return arcs


def navsol_line(r: Dict[str, Any]) -> str:
    t = r["dt_utc"]
    prn_str = " ".join(f"{p:02d}" for p in r["prns"])
    line = (
        f"001 {t.year:04d} {t.month:02d} {t.day:02d} "
        f"{t.hour:02d} {t.minute:02d} {secms5(t)} "
        f"{int(r['valid'])}  {int(r['mode'])} "
        f"{r['x_m']:.12e}  {r['y_m']:.12e}  {r['z_m']:.12e} "
        f"0 {int(r['nsv'])}"
    )
    if prn_str:
        line += " " + prn_str
    return line


def write_navsol(rows: List[Dict[str, Any]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(navsol_line(r) + "\n")


def write_csv(rows: List[Dict[str, Any]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["iso_utc","x_m","y_m","z_m","valid","mode","nsv","prns","source","gps_week","gps_sow","solstat","postype"])
        for r in rows:
            iso = r["dt_utc"].strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            prns = ",".join(str(p) for p in r["prns"])
            w.writerow([iso, r["x_m"], r["y_m"], r["z_m"], r["valid"], r["mode"], r["nsv"], prns,
                        r["source"], r["gps_week"], r["gps_sow"], r["solstat"], r["postype"]])


def group_rows_by_utc_date(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    rows를 UTC 날짜(YYYY-MM-DD)별로 그룹화
    """
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        day = r["dt_utc"].strftime("%Y-%m-%d")
        groups[day].append(r)
    for day in groups:
        groups[day].sort(key=lambda x: x["dt_utc"])
    return dict(groups)


def write_daily_outputs(
    rows: List[Dict[str, Any]],
    out_dir: str,
    write_csv_too: bool = True,
    also_save_longest: bool = False,
    max_gap_s: float = 60.0,          # 연속성 판단 기준(기본값 60초), 만약 GPS수신기 데이터 수집 gap +60초 이상 변경 시 해당 수치 조정해야함
) -> List[str]:
    """
    날짜별로 navsol_YYYY-MM-DD.navSol (+ csv) 저장
    옵션: also_save_longest=True면, 가장 긴 연속 arc를 navsol_YYYY-MM-DD_longest.* 로 추가 저장
    """
    os.makedirs(out_dir, exist_ok=True)
    created: List[str] = []
    groups = group_rows_by_utc_date(rows)

    def arc_duration_s(arc: List[Dict[str, Any]]) -> float:
        if len(arc) < 2:
            return 0.0
        return (arc[-1]["dt_utc"] - arc[0]["dt_utc"]).total_seconds()

    for day, day_rows in sorted(groups.items()):
        # full day 저장
        nav_path = os.path.join(out_dir, f"navsol_{day}.navSol")
        write_navsol(day_rows, nav_path)
        created.append(nav_path)

        if write_csv_too:
            csv_path = os.path.join(out_dir, f"navsol_{day}.csv")
            write_csv(day_rows, csv_path)
            created.append(csv_path)

        # longest arc 추가 저장
        if also_save_longest:
            arcs = split_arcs(day_rows, max_gap_s=max_gap_s)
            if arcs:
                longest = max(arcs, key=arc_duration_s)

                nav_path2 = os.path.join(out_dir, f"navsol_{day}_longest.navSol")
                write_navsol(longest, nav_path2)
                created.append(nav_path2)

                if write_csv_too:
                    csv_path2 = os.path.join(out_dir, f"navsol_{day}_longest.csv")
                    write_csv(longest, csv_path2)
                    created.append(csv_path2)

    return created


def gui_main():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("NavSol Extractor (GNSS Log -> navSol/CSV)")
    root.geometry("720x420")

    files_var = tk.StringVar(value="선택된 파일 없음")
    outdir_var = tk.StringVar(value=os.path.join(os.getcwd(), "output"))
    leap_var = tk.IntVar(value=18)
    valid_only_var = tk.BooleanVar(value=True)
    write_csv_var = tk.BooleanVar(value=True)
    also_longest_var = tk.BooleanVar(value=True)
    max_gap_var = tk.DoubleVar(value=60.0)

    selected_files: List[str] = []

    def refresh_files_label():
        if not selected_files:
            files_var.set("선택된 파일 없음")
        else:
            files_var.set(f"{len(selected_files)}개 파일 선택됨")

    def on_add_files():
        nonlocal selected_files
        files = filedialog.askopenfilenames(
            title="GNSS Raw Log 파일 선택(.dat/.txt)",
            filetypes=[("Log files", "*.dat *.txt *.log"), ("All files", "*.*")]
        )
        if files:
            # 중복 제거
            merged = list(dict.fromkeys(list(selected_files) + list(files)))
            selected_files = merged
            refresh_files_label()

    def on_clear_files():
        nonlocal selected_files
        selected_files = []
        refresh_files_label()

    def on_pick_outdir():
        d = filedialog.askdirectory(title="출력 폴더 선택")
        if d:
            outdir_var.set(d)

    def on_run():
        if not selected_files:
            messagebox.showwarning("경고", "먼저 GNSS 로그 파일을 선택하세요.")
            return

        out_dir = outdir_var.get().strip()
        if not out_dir:
            messagebox.showwarning("경고", "출력 폴더를 지정하세요.")
            return

        try:
            # 추출
            rows = extract_records(
                paths=list(selected_files),
                leap_seconds=int(leap_var.get()),
                valid_only=bool(valid_only_var.get())
            )

            if not rows:
                messagebox.showinfo("결과", "추출된 NavSol 레코드가 없습니다.")
                return

            created = write_daily_outputs(
                rows, 
                out_dir, 
                write_csv_too=bool(write_csv_var.get()),
                also_save_longest=bool(also_longest_var.get()),
                max_gap_s=float(max_gap_var.get()),
            )

            # 결과 요약
            groups = group_rows_by_utc_date(rows)
            summary_lines = [f"입력 파일: {len(selected_files)}개",
                             f"추출 레코드: {len(rows)}개",
                             f"생성 날짜: {len(groups)}일",
                             f"출력 폴더: {out_dir}",
                             "",
                             "날짜별 레코드 수:"]
            for day, day_rows in sorted(groups.items()):
                summary_lines.append(f"  - {day}: {len(day_rows)}")

            messagebox.showinfo("완료", "\n".join(summary_lines))

        except Exception as e:
            messagebox.showerror("오류", str(e))

    # --- UI Layout ---
    frm = ttk.Frame(root, padding=10)
    frm.pack(fill="both", expand=True)

    row = 0
    ttk.Label(frm, text="1) GNSS 로그 파일 선택").grid(row=row, column=0, sticky="w")
    ttk.Button(frm, text="파일 추가", command=on_add_files).grid(row=row, column=1, sticky="w", padx=5)
    ttk.Button(frm, text="선택 초기화", command=on_clear_files).grid(row=row, column=2, sticky="w")
    row += 1

    ttk.Label(frm, textvariable=files_var).grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 10))
    row += 1

    ttk.Label(frm, text="2) 출력 폴더").grid(row=row, column=0, sticky="w")
    ttk.Entry(frm, textvariable=outdir_var, width=60).grid(row=row, column=1, sticky="we", padx=5)
    ttk.Button(frm, text="찾기", command=on_pick_outdir).grid(row=row, column=2, sticky="w")
    row += 1

    ttk.Label(frm, text="3) 옵션").grid(row=row, column=0, sticky="w", pady=(10, 0))
    row += 1

    opt = ttk.Frame(frm)
    opt.grid(row=row, column=0, columnspan=3, sticky="we")
    ttk.Label(opt, text="GPS-UTC leap seconds").grid(row=0, column=0, sticky="w")
    ttk.Spinbox(opt, from_=0, to=30, textvariable=leap_var, width=5).grid(row=0, column=1, sticky="w", padx=5)
    ttk.Checkbutton(opt, text="SOL_COMPUTED (valid only)", variable=valid_only_var).grid(row=0, column=2, sticky="w", padx=10)
    ttk.Checkbutton(opt, text="CSV 저장", variable=write_csv_var).grid(row=0, column=3, sticky="w", padx=10)
    ttk.Checkbutton(opt, text="Longest arc", variable=also_longest_var).grid(row=1, column=0, sticky="w", pady=(6,0))
    ttk.Label(opt, text="gap(s)").grid(row=1, column=1, sticky="e", pady=(6,0))
    ttk.Spinbox(opt, from_=1, to=600, textvariable=max_gap_var, width=8).grid(row=1, column=2, sticky="w", padx=5, pady=(6,0))
    row += 1

    ttk.Separator(frm).grid(row=row, column=0, columnspan=3, sticky="we", pady=15)
    row += 1

    ttk.Button(frm, text="실행", command=on_run).grid(row=row, column=0, sticky="w")
    ttk.Label(frm, text="(날짜별로 navsol_YYYY-MM-DD.navSol / navsol_YYYY-MM-DD.csv 저장)").grid(row=row, column=1, columnspan=2, sticky="w")

    frm.columnconfigure(1, weight=1)

    refresh_files_label()
    root.mainloop()


def pick_files_dialog() -> List[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        files = filedialog.askopenfilenames(title="Select raw GNSS log files (.dat/.txt)")
        root.destroy()
        return list(files)
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", nargs="*", help="Input raw log paths or globs (e.g., 20250502_*.dat)")
    ap.add_argument("--pick", action="store_true", help="Open a file dialog to pick input files")
    ap.add_argument("--out", required=True, help="Output .navSol path")
    ap.add_argument("--csv", dest="csv_out", default=None, help="Optional output CSV path")
    ap.add_argument("--leap-seconds", type=int, default=18, help="GPS-UTC leap seconds (default: 18)")
    ap.add_argument("--valid-only", action="store_true", help="Keep only SOL_COMPUTED solutions")
    ap.add_argument("--max-gap", type=float, default=1.5, help="Arc split gap threshold in seconds (default: 1.5)")
    ap.add_argument("--min-arc", type=float, default=0.0, help="Keep only arcs with duration >= this seconds (default: 0)")
    ap.add_argument("--split-arcs", action="store_true", help="Write each arc to a separate file with suffix _arcNNN")

    args = ap.parse_args()

    paths: List[str] = []
    if args.pick:
        paths.extend(pick_files_dialog())

    if args.inputs:
        for token in args.inputs:
            expanded = glob.glob(token)
            if expanded:
                paths.extend(expanded)
            else:
                paths.append(token)

    paths = [p for p in paths if p and os.path.exists(p)]
    paths = sorted(set(paths))

    if not paths:
        raise SystemExit("No input files found. Use --in or --pick.")

    rows = extract_records(paths, leap_seconds=args.leap_seconds, valid_only=args.valid_only)

    # Optional arc filtering
    arcs = split_arcs(rows, max_gap_s=args.max_gap) if args.min_arc > 0 or args.split_arcs else [rows]
    if arcs and arcs[0] is rows:
        # Not split; treat as single arc for filtering
        arcs = [rows]

    kept: List[Dict[str, Any]] = []
    out_base, out_ext = os.path.splitext(args.out)

    def arc_duration_s(arc: List[Dict[str, Any]]) -> float:
        if len(arc) < 2:
            return 0.0
        return (arc[-1]["dt_utc"] - arc[0]["dt_utc"]).total_seconds()

    if args.split_arcs:
        idx = 0
        for arc in split_arcs(rows, max_gap_s=args.max_gap):
            if arc_duration_s(arc) < args.min_arc:
                continue
            idx += 1
            out_path = f"{out_base}_arc{idx:03d}{out_ext}"
            write_navsol(arc, out_path)
        # For CSV, write all kept records
        if args.csv_out:
            for arc in split_arcs(rows, max_gap_s=args.max_gap):
                if arc_duration_s(arc) >= args.min_arc:
                    kept.extend(arc)
            write_csv(kept, args.csv_out)
    else:
        # Keep arcs >= min_arc but write into one file (sorted)
        if args.min_arc > 0:
            for arc in split_arcs(rows, max_gap_s=args.max_gap):
                if arc_duration_s(arc) >= args.min_arc:
                    kept.extend(arc)
            kept.sort(key=lambda r: r["dt_utc"])
        else:
            kept = rows
        write_navsol(kept, args.out)
        if args.csv_out:
            write_csv(kept, args.csv_out)

    print(f"Inputs: {len(paths)} file(s)")
    print(f"Records extracted: {len(rows)}")
    print(f"Output: {args.out}")
    if args.csv_out:
        print(f"CSV: {args.csv_out}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        gui_main()
    else:
        main()