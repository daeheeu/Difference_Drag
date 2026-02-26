from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.tools.plot_case_a outputs/case_a_1sat.csv")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    # time index (row number) is fine for quick-look plots
    x = range(len(df))

    out_dir = csv_path.parent
    stem = csv_path.stem

    # semi-major axis
    plt.figure()
    plt.plot(x, df["a_m"])
    plt.xlabel("step")
    plt.ylabel("a (m)")
    plt.title("Semi-major axis vs step")
    plt.tight_layout()
    plt.savefig(out_dir / f"{stem}_a.png", dpi=150)
    plt.close()

    # density
    plt.figure()
    plt.plot(x, df["rho_kgm3"])
    plt.xlabel("step")
    plt.ylabel("rho (kg/m^3)")
    plt.title("Density vs step")
    plt.tight_layout()
    plt.savefig(out_dir / f"{stem}_rho.png", dpi=150)
    plt.close()

    # altitude (optional quick look)
    if "altitude_m" in df.columns:
        plt.figure()
        plt.plot(x, df["altitude_m"])
        plt.xlabel("step")
        plt.ylabel("altitude (m)")
        plt.title("Geodetic altitude vs step")
        plt.tight_layout()
        plt.savefig(out_dir / f"{stem}_alt.png", dpi=150)
        plt.close()

    print(f"[OK] Saved plots to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()