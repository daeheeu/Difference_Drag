# src/pipelines/ccsds_oem.py
from __future__ import annotations

"""
CCSDS OEM (Orbit Ephemeris Message) writer.

Interoperability note:
- Many tools (STK/GMAT and various OEM parsers) assume OEM state vectors are in **km** and **km/s**.
- This project internally uses meters (m) and meters per second (m/s) in CSV artifacts.
- Therefore, this writer outputs OEM states in **km** and **km/s** by default, while leaving CSV outputs untouched.

If you need meters in OEM for a specific consumer, set units="m".
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd


@dataclass
class OEMMeta:
    object_name: str = "SAT"
    object_id: str = "UNKNOWN"
    center_name: str = "EARTH"
    ref_frame: str = "GCRF"   # e.g., GCRF / EME2000
    time_system: str = "UTC"
    originator: str = "Difference_Drag"


def write_oem_from_ephem_df(
    df: pd.DataFrame,
    out_path: str | Path,
    meta: OEMMeta,
    time_col: str = "iso_utc",
    x: str = "x_i_m",
    y: str = "y_i_m",
    z: str = "z_i_m",
    vx: str = "vx_i_mps",
    vy: str = "vy_i_mps",
    vz: str = "vz_i_mps",
    units: str = "km",  # "km" (default) or "m"
) -> None:
    """
    Write a CCSDS OEM text file.

    Parameters
    ----------
    df : pd.DataFrame
        Ephemeris dataframe with ISO UTC time column and inertial PV columns.
        Internal convention: meters and m/s.
    units : str
        "km" (default): output positions in km and velocities in km/s.
        "m": output positions in m and velocities in m/s.
    """
    out_path = Path(out_path)
    if df is None or df.empty:
        raise ValueError("Ephemeris dataframe is empty.")

    units = units.lower().strip()
    if units not in ("km", "m"):
        raise ValueError(f"units must be 'km' or 'm', got: {units!r}")

    # Ensure sorted
    df = df.sort_values(time_col).reset_index(drop=True)

    creation = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    t_start = str(df.loc[0, time_col])
    t_stop = str(df.loc[len(df) - 1, time_col])

    # Scaling
    if units == "km":
        pos_scale = 1.0 / 1000.0
        vel_scale = 1.0 / 1000.0
        unit_tag = "km km/s"
    else:
        pos_scale = 1.0
        vel_scale = 1.0
        unit_tag = "m m/s"

    lines: list[str] = []
    lines.append("CCSDS_OEM_VERS = 2.0")
    lines.append(f"CREATION_DATE = {creation}")
    lines.append(f"ORIGINATOR = {meta.originator}")
    lines.append("")
    lines.append("META_START")
    lines.append(f"OBJECT_NAME = {meta.object_name}")
    lines.append(f"OBJECT_ID = {meta.object_id}")
    lines.append(f"CENTER_NAME = {meta.center_name}")
    lines.append(f"REF_FRAME = {meta.ref_frame}")
    lines.append(f"TIME_SYSTEM = {meta.time_system}")
    lines.append(f"START_TIME = {t_start}")
    lines.append(f"STOP_TIME = {t_stop}")
    lines.append("META_STOP")
    lines.append("")
    lines.append(f"# ISO_TIME X Y Z VX VY VZ  (units: {unit_tag})")

    for _, r in df.iterrows():
        X = float(r[x]) * pos_scale
        Y = float(r[y]) * pos_scale
        Z = float(r[z]) * pos_scale
        VX = float(r[vx]) * vel_scale
        VY = float(r[vy]) * vel_scale
        VZ = float(r[vz]) * vel_scale

        lines.append(
            f"{r[time_col]} "
            f"{X:.9f} {Y:.9f} {Z:.9f} "
            f"{VX:.12f} {VY:.12f} {VZ:.12f}"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
