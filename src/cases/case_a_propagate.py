from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.orekit_bootstrap import init_orekit


@dataclass
class CaseAConfig:
    epoch_utc: str
    duration_sec: int
    step_sec: int
    altitude_m: float
    inc_deg: float
    raan_deg: float
    argp_deg: float
    m0_deg: float
    mass_kg: float
    cd: float
    area_m2: float
    degree: int
    order: int
    csv_path: str


def parse_config(path: str) -> CaseAConfig:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    oi = cfg["orbit_init"]
    sc = cfg["spacecraft"]
    gr = cfg["gravity"]
    out = cfg["outputs"]

    return CaseAConfig(
        epoch_utc=cfg["epoch_utc"],
        duration_sec=int(cfg["duration_sec"]),
        step_sec=int(cfg["step_sec"]),
        altitude_m=float(oi["altitude_m"]),
        inc_deg=float(oi["inclination_deg"]),
        raan_deg=float(oi["raan_deg"]),
        argp_deg=float(oi["argp_deg"]),
        m0_deg=float(oi["mean_anomaly_deg"]),
        mass_kg=float(sc["mass_kg"]),
        cd=float(sc["cd"]),
        area_m2=float(sc["area_m2"]),
        degree=int(gr["degree"]),
        order=int(gr["order"]),
        csv_path=str(out["csv_path"]),
    )


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.cases.case_a_propagate <config.json>")
        sys.exit(1)

    config_path = sys.argv[1]
    cfg = parse_config(config_path)

    # Start JVM + set classpath + load orekit-data
    init_orekit()

    # JPype interface implementation helpers
    from jpype import JImplements, JOverride

    # Orekit / Hipparchus imports (must be AFTER init_orekit)
    from org.orekit.orbits import OrbitType
    from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
    from org.orekit.time import AbsoluteDate, TimeScalesFactory
    from org.orekit.frames import FramesFactory
    from org.orekit.utils import IERSConventions, Constants
    from org.orekit.orbits import KeplerianOrbit, PositionAngleType
    from org.orekit.propagation.numerical import NumericalPropagator
    from org.orekit.propagation import SpacecraftState
    from org.orekit.forces.gravity.potential import GravityFieldFactory
    from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel
    from org.orekit.bodies import OneAxisEllipsoid
    from org.orekit.forces.drag import DragForce, IsotropicDrag
    from org.orekit.models.earth.atmosphere import SimpleExponentialAtmosphere

    def utc_date(iso_z: str) -> AbsoluteDate:
        return AbsoluteDate(iso_z.replace("Z", ""), TimeScalesFactory.getUTC())

    # Frames / Earth
    utc = TimeScalesFactory.getUTC()
    inertial = FramesFactory.getEME2000()
    itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)

    earth = OneAxisEllipsoid(
        Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
        Constants.WGS84_EARTH_FLATTENING,
        itrf,
    )

    # Initial orbit
    mu = Constants.EGM96_EARTH_MU
    a0 = Constants.WGS84_EARTH_EQUATORIAL_RADIUS + cfg.altitude_m
    e0 = 0.0
    i0 = math.radians(cfg.inc_deg)
    raan0 = math.radians(cfg.raan_deg)
    argp0 = math.radians(cfg.argp_deg)
    m0 = math.radians(cfg.m0_deg)
    date0 = utc_date(cfg.epoch_utc)

    orbit0 = KeplerianOrbit(
        a0,
        e0,
        i0,
        argp0,
        raan0,
        m0,
        PositionAngleType.MEAN,
        inertial,
        date0,
        mu,
    )

    # Integrator / propagator
    min_step = 1.0
    max_step = float(cfg.step_sec)
    pos_tol = 10.0
    integrator = DormandPrince853Integrator(min_step, max_step, pos_tol, pos_tol)

    propagator = NumericalPropagator(integrator)
    propagator.setInitialState(SpacecraftState(orbit0, cfg.mass_kg))

    # Gravity
    provider = GravityFieldFactory.getNormalizedProvider(cfg.degree, cfg.order)
    propagator.addForceModel(HolmesFeatherstoneAttractionModel(itrf, provider))

    # Atmosphere (MVP exponential)
    rho0 = 3.614e-13
    h0 = 500000.0
    h_scale = 60000.0
    atmosphere = SimpleExponentialAtmosphere(earth, rho0, h0, h_scale)

    drag = DragForce(atmosphere, IsotropicDrag(cfg.area_m2, cfg.cd))
    propagator.addForceModel(drag)

    rows = []

    @JImplements("org.orekit.propagation.sampling.OrekitFixedStepHandler")
    class StepHandler:
        def __init__(self, earth_obj, atm_obj, utc_scale, rows_ref):
            self.earth = earth_obj
            self.atm = atm_obj
            self.utc = utc_scale
            self.rows = rows_ref

        @JOverride
        def init(self, s0, t, step):
            # s0: SpacecraftState, t: AbsoluteDate, step: double
            return

        @JOverride
        def handleStep(self, currentState):
            date = currentState.getDate()
            orbit = currentState.getOrbit()
            pv = currentState.getPVCoordinates()
            r = pv.getPosition()
            v = pv.getVelocity()

            geo = self.earth.transform(r, currentState.getFrame(), date)
            alt = geo.getAltitude()
            rho = self.atm.getDensity(date, r, currentState.getFrame())

            # Convert to Keplerian for RAAN/argp/M outputs
            kep = OrbitType.KEPLERIAN.convertType(orbit)

            self.rows.append(
                {
                    "t_utc": date.toString(self.utc),
                    "a_m": float(kep.getA()),
                    "altitude_m": float(alt),
                    "e": float(kep.getE()),
                    "i_deg": math.degrees(float(kep.getI())),
                    "raan_deg": math.degrees(float(kep.getRightAscensionOfAscendingNode())),
                    "argp_deg": math.degrees(float(kep.getPerigeeArgument())),
                    "M_deg": math.degrees(float(kep.getMeanAnomaly())),
                    "rho_kgm3": float(rho),
                    "rx_m": float(r.getX()),
                    "ry_m": float(r.getY()),
                    "rz_m": float(r.getZ()),
                    "vx_ms": float(v.getX()),
                    "vy_ms": float(v.getY()),
                    "vz_ms": float(v.getZ()),
                }
            )

        # Orekit 13.x may call finish() at the end of propagation
        @JOverride
        def finish(self, finalState):
            # finalState: SpacecraftState
            return

    handler = StepHandler(earth, atmosphere, utc, rows)
    propagator.setStepHandler(float(cfg.step_sec), handler)

    end_date = date0.shiftedBy(float(cfg.duration_sec))
    propagator.propagate(end_date)

    out_path = Path(cfg.csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Wrote: {out_path.resolve()} rows={len(rows)}")


if __name__ == "__main__":
    main()