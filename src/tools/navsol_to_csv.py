from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def convert(navsol_path: Path, out_csv: Path, valid_only: bool = True):
    rows = []
    with navsol_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            t = line.strip().split()
            if len(t) < 14:
                continue

            year, mon, day = int(t[1]), int(t[2]), int(t[3])
            hh, mm = int(t[4]), int(t[5])
            msec = int(t[6])  # 예: 46000 = 46.000 sec
            sec = msec / 1000.0

            status = int(t[7])   # 1=good, 0=bad(추정 실패/저품질 가능)
            mode = int(t[8])     # 보통 2

            if valid_only and status != 1:
                continue

            x, y, z = float(t[9]), float(t[10]), float(t[11])
            nsv = int(t[13])
            prns = " ".join(t[14:14+nsv]) if len(t) >= 14+nsv else ""

            iso_utc = f"{year:04d}-{mon:02d}-{day:02d}T{hh:02d}:{mm:02d}:{sec:06.3f}Z"

            rows.append({
                "iso_utc": iso_utc,
                "x_m": x, "y_m": y, "z_m": z,
                "valid": 1 if status == 1 else 0,
                "mode": mode,
                "nsv": nsv,
                "prns": prns,
                "source": "FDS_NAVSOL",
            })

    if not rows:
        raise RuntimeError("No rows parsed. Check file format or valid_only setting.")

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_csv} rows={len(df)} (valid_only={valid_only})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("navsol", type=str)
    ap.add_argument("out_csv", type=str)
    ap.add_argument("--all", action="store_true", help="include status=0 rows too")
    args = ap.parse_args()
    convert(Path(args.navsol), Path(args.out_csv), valid_only=(not args.all))

if __name__ == "__main__":
    main()