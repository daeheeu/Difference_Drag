from __future__ import annotations

import json
import sys
import math
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

from src.orekit_bootstrap import init_orekit


@dataclass
class CaseBEclipseConfig:
    epoch_utc: str
    duration_sec: int
    out_csv: str


def parse_config(path: str) -> CaseBEclipseConfig:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    epoch_utc = cfg["epoch_utc"]
    duration_sec = int(cfg["duration_sec"])
    out_csv = cfg.get("outputs", {}).get("eclipse_csv_path", "outputs/case_b_eclipse.csv")
    return CaseBEclipseConfig(epoch_utc=epoch_utc, duration_sec=duration_sec, out_csv=out_csv)


def fmt_utc_ms(date_obj, utc_scale) -> str:
    """
    Orekit AbsoluteDate -> 'YYYY-MM-DDTHH:MM:SS.mmm' (UTC)
    """
    s = str(date_obj.toString(utc_scale)).strip()
    if "." not in s:
        return s + ".000"
    head, frac = s.split(".", 1)
    ms = (frac + "000")[:3]
    return f"{head}.{ms}"


def parse_utc_ms(s: str) -> datetime:
    """
    'YYYY-MM-DDTHH:MM:SS.mmm' 또는 'YYYY-MM-DDTHH:MM:SS' -> datetime(UTC)
    """
    if "." not in s:
        s = s + ".000"
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%f").replace(tzinfo=timezone.utc)


def fmt_mmss_ss(seconds: float) -> str:
    """
    seconds -> 'MM:SS.ss'
    """
    if seconds < 0:
        seconds = 0.0
    m = int(seconds // 60)
    s = seconds - 60 * m
    return f"{m:02d}:{s:05.2f}"


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.cases.case_b_eclipse_timeline <case_a_config.json>")
        sys.exit(1)

    config_path = sys.argv[1]
    cfg = parse_config(config_path)

    init_orekit()

    # Java imports after init
    from org.orekit.time import AbsoluteDate, TimeScalesFactory
    from org.orekit.frames import FramesFactory
    from org.orekit.utils import IERSConventions, Constants
    from org.orekit.orbits import KeplerianOrbit, PositionAngleType
    from org.orekit.propagation.numerical import NumericalPropagator
    from org.orekit.propagation import SpacecraftState
    from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
    from org.orekit.forces.gravity.potential import GravityFieldFactory
    from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel
    from org.orekit.bodies import OneAxisEllipsoid, CelestialBodyFactory
    from org.orekit.propagation.events import EclipseDetector
    from org.orekit.propagation.events.handlers import ContinueOnEvent
    from org.orekit.propagation.events import EventsLogger

    # Read orbit_init from the same config
    cfg_all = json.loads(Path(config_path).read_text(encoding="utf-8"))
    oi = cfg_all["orbit_init"]
    altitude_m = float(oi["altitude_m"])
    inc_deg = float(oi["inclination_deg"])
    raan_deg = float(oi["raan_deg"])
    argp_deg = float(oi["argp_deg"])
    m0_deg = float(oi["mean_anomaly_deg"])

    # Time/frames
    utc = TimeScalesFactory.getUTC()
    inertial = FramesFactory.getEME2000()
    itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)

    date0 = AbsoluteDate(cfg.epoch_utc.replace("Z", ""), utc)
    end_date = date0.shiftedBy(float(cfg.duration_sec))

    # Simple initial orbit
    mu = Constants.EGM96_EARTH_MU
    a0 = Constants.WGS84_EARTH_EQUATORIAL_RADIUS + altitude_m

    orbit0 = KeplerianOrbit(
        a0,
        0.0,
        math.radians(inc_deg),
        math.radians(argp_deg),
        math.radians(raan_deg),
        math.radians(m0_deg),
        PositionAngleType.MEAN,
        inertial,
        date0,
        mu,
    )

    # Propagator (gravity only is enough for eclipse timing)
    min_step = 1.0
    max_step = 60.0
    pos_tol = 10.0
    integrator = DormandPrince853Integrator(min_step, max_step, pos_tol, pos_tol)
    propagator = NumericalPropagator(integrator)
    propagator.setInitialState(SpacecraftState(orbit0, 1.0))

    provider = GravityFieldFactory.getNormalizedProvider(20, 20)
    propagator.addForceModel(HolmesFeatherstoneAttractionModel(itrf, provider))

    # Earth & Sun
    earth = OneAxisEllipsoid(
        Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
        Constants.WGS84_EARTH_FLATTENING,
        itrf,
    )
    sun = CelestialBodyFactory.getSun()

    # Eclipse detectors
    umbra = EclipseDetector(sun, Constants.SUN_RADIUS, earth).withHandler(ContinueOnEvent())
    penumbra = EclipseDetector(sun, Constants.SUN_RADIUS, earth).withPenumbra().withHandler(ContinueOnEvent())

    logger = EventsLogger()
    propagator.addEventDetector(logger.monitorDetector(umbra))
    propagator.addEventDetector(logger.monitorDetector(penumbra))

    propagator.propagate(end_date)

    # Build events + durations (duration only on IN rows)
    events = logger.getLoggedEvents()
    raw_rows = []
    for ev in events:
        det = ev.getEventDetector()
        date = ev.getState().getDate()
        increasing = ev.isIncreasing()

        if det == umbra:
            kind = "UMBRA_OUT" if increasing else "UMBRA_IN"
            key = "UMBRA"
        else:
            kind = "PENUMBRA_OUT" if increasing else "PENUMBRA_IN"
            key = "PENUMBRA"

        t_str = fmt_utc_ms(date, utc)
        raw_rows.append({"t_utc": t_str, "event": kind, "key": key})

    # Sort by time
    raw_rows.sort(key=lambda r: r["t_utc"])

    # Fill duration on IN rows by pairing with next OUT
    last_in_time = {"PENUMBRA": None, "UMBRA": None}
    output_rows = []

    for r in raw_rows:
        t_str = r["t_utc"]
        ev = r["event"]
        key = r["key"]

        # default: empty duration
        output_rows.append({"t_utc": t_str, "event": ev, "Duration": ""})

        if ev.endswith("_IN"):
            last_in_time[key] = parse_utc_ms(t_str)

        elif ev.endswith("_OUT") and last_in_time[key] is not None:
            tout = parse_utc_ms(t_str)
            dur_sec = (tout - last_in_time[key]).total_seconds()
            last_in_time[key] = None

            # find most recent matching IN row and fill duration there
            in_tag = f"{key}_IN"
            for j in range(len(output_rows) - 1, -1, -1):
                if output_rows[j]["event"] == in_tag and output_rows[j]["Duration"] == "":
                    output_rows[j]["Duration"] = fmt_mmss_ss(dur_sec)
                    break

    # Save
    df = pd.DataFrame(output_rows)
    out_path = Path(cfg.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Wrote: {out_path.resolve()} rows={len(df)}")


if __name__ == "__main__":
    main()