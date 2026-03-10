from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Optional


@dataclass
class ForceModelCfg:
    # profile switch
    profile: str = "legacy"

    # spacecraft properties
    mass_kg: float = 30.0
    area_m2: float = 0.052578
    cd0: float = 2.3

    # phase-1/2 policy
    estimate_cd: bool = False

    # gravity / third body
    gravity_degree: int = 20
    gravity_order: int = 20
    use_third_body: bool = True

    # drag
    drag_enabled: bool = True
    atmosphere: str = "SIMPLE_EXP"
    rho0: float = 3.614e-13
    h0_m: float = 500000.0
    h_scale_m: float = 60000.0

    # SRP
    use_srp: bool = False
    cr0: float = 1.5
    srp_area_m2: Optional[float] = None

    # tides
    use_tides: bool = False
    tide_system: Optional[str] = None
    use_ocean_tides: bool = False
    ocean_tides_degree: int = 4
    ocean_tides_order: int = 4


@dataclass
class BuiltForceModelBundle:
    models: List[Any]
    drag_sensitive: Optional[Any] = None
    radiation_sensitive: Optional[Any] = None
    requested_atmosphere: Optional[str] = None
    realized_atmosphere: Optional[str] = None
    notes: Optional[List[str]] = None


def force_cfg_from_dict(raw: Optional[Dict[str, Any]]) -> ForceModelCfg:
    raw = dict(raw or {})

    allowed = {f.name for f in fields(ForceModelCfg)}
    filtered = {k: v for k, v in raw.items() if k in allowed}

    cfg = ForceModelCfg(**filtered)

    profile = (cfg.profile or "legacy").strip().lower()

    if profile == "microcosm_like":
        if "gravity_degree" not in filtered:
            cfg.gravity_degree = 70
        if "gravity_order" not in filtered:
            cfg.gravity_order = 70
        if "use_third_body" not in filtered:
            cfg.use_third_body = True
        if "drag_enabled" not in filtered:
            cfg.drag_enabled = True
        if "use_srp" not in filtered:
            cfg.use_srp = True
        if "use_tides" not in filtered:
            cfg.use_tides = True
        if "tide_system" not in filtered:
            cfg.tide_system = "ZERO_TIDE"
        if "cr0" not in filtered:
            cfg.cr0 = 1.5
        if "srp_area_m2" not in filtered:
            cfg.srp_area_m2 = cfg.area_m2

    return cfg


def _resolve_tide_system(name: Optional[str]):
    from org.orekit.forces.gravity.potential import TideSystem

    if name is None:
        return None

    key = str(name).strip().upper()

    if key == "ZERO_TIDE":
        return TideSystem.ZERO_TIDE
    if key == "TIDE_FREE":
        return TideSystem.TIDE_FREE

    raise ValueError(f"Unsupported tide_system: {name}")


def force_cfg_to_dict(
    cfg: ForceModelCfg,
    bundle: Optional[BuiltForceModelBundle] = None,
) -> Dict[str, Any]:
    out = asdict(cfg)
    if bundle is not None:
        out["requested_atmosphere"] = bundle.requested_atmosphere
        out["realized_atmosphere"] = bundle.realized_atmosphere
        out["force_model_notes"] = list(bundle.notes or [])
    return out


def build_force_model_bundle(*, itrf: Any, earth: Any, forces: ForceModelCfg) -> BuiltForceModelBundle:
    """
    Build Orekit force-model objects, but do NOT attach them yet.
    This is shared by both OD and OP.
    """

    # Orekit imports AFTER JVM startup
    from org.orekit.bodies import CelestialBodyFactory
    from org.orekit.forces.drag import DragForce, IsotropicDrag
    from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel, ThirdBodyAttraction, SolidTides
    from org.orekit.forces.gravity.potential import GravityFieldFactory, TideSystem
    from org.orekit.forces.radiation import SolarRadiationPressure, IsotropicRadiationSingleCoefficient
    from org.orekit.models.earth.atmosphere import SimpleExponentialAtmosphere
    from org.orekit.time import TimeScalesFactory
    from org.orekit.utils import Constants, IERSConventions

    models: List[Any] = []
    notes: List[str] = []

    # -------------------------
    # Gravity field
    # -------------------------
    grav = GravityFieldFactory.getNormalizedProvider(
        int(forces.gravity_degree),
        int(forces.gravity_order),
    )
    models.append(HolmesFeatherstoneAttractionModel(itrf, grav))

    # -------------------------
    # 3rd bodies
    # -------------------------
    sun = CelestialBodyFactory.getSun()
    moon = CelestialBodyFactory.getMoon()

    if bool(forces.use_third_body):
        models.append(ThirdBodyAttraction(sun))
        models.append(ThirdBodyAttraction(moon))

    # -------------------------
    # Drag
    # -------------------------
    drag_sensitive = None
    requested_atm = str(forces.atmosphere or "SIMPLE_EXP").strip().upper()
    realized_atm = None

    if bool(forces.drag_enabled):
        if requested_atm == "SIMPLE_EXP":
            atmosphere = SimpleExponentialAtmosphere(
                earth,
                float(forces.rho0),
                float(forces.h0_m),
                float(forces.h_scale_m),
            )
            realized_atm = "SIMPLE_EXP"
        else:
            atmosphere = SimpleExponentialAtmosphere(
                earth,
                float(forces.rho0),
                float(forces.h0_m),
                float(forces.h_scale_m),
            )
            realized_atm = "SIMPLE_EXP"
            notes.append(
                f"Atmosphere '{requested_atm}' is not implemented yet; "
                "falling back to SIMPLE_EXP."
            )

        drag_sensitive = IsotropicDrag(float(forces.area_m2), float(forces.cd0))
        models.append(DragForce(atmosphere, drag_sensitive))

    # -------------------------
    # SRP
    # -------------------------
    radiation_sensitive = None
    if bool(forces.use_srp):
        srp_area = float(forces.srp_area_m2) if forces.srp_area_m2 is not None else float(forces.area_m2)
        radiation_sensitive = IsotropicRadiationSingleCoefficient(srp_area, float(forces.cr0))
        models.append(SolarRadiationPressure(sun, earth, radiation_sensitive))
        notes.append(f"SRP enabled with area={srp_area} m^2, Cr={float(forces.cr0)}")

    # -------------------------
    # Solid tides
    # -------------------------
    if bool(forces.use_tides):
        ut1 = TimeScalesFactory.getUT1(IERSConventions.IERS_2010, True)

        forced_tide_system = _resolve_tide_system(forces.tide_system)

        if forced_tide_system is not None:
            tide_system = forced_tide_system
            notes.append(
                f"Solid tides enabled with forced tide_system={str(tide_system)}"
            )
        else:
            try:
                tide_system = grav.getTideSystem()
                notes.append(
                    f"Solid tides enabled with provider tide_system={str(tide_system)}"
                )
            except Exception:
                tide_system = TideSystem.ZERO_TIDE
                notes.append(
                    "Gravity provider tide system not available; using fallback tide_system=ZERO_TIDE"
                )

        models.append(
            SolidTides(
                itrf,
                Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
                Constants.WGS84_EARTH_MU,
                tide_system,
                IERSConventions.IERS_2010,
                ut1,
                sun,
                moon,
            )
        )
        notes.append(f"Solid tides enabled with tide_system={str(tide_system)}")

    # -------------------------
    # Ocean tides (parked for now)
    # -------------------------
    if bool(forces.use_ocean_tides):
        notes.append(
            "use_ocean_tides=true requested, but OceanTides is intentionally parked in step-3."
        )

    return BuiltForceModelBundle(
        models=models,
        drag_sensitive=drag_sensitive,
        radiation_sensitive=radiation_sensitive,
        requested_atmosphere=requested_atm,
        realized_atmosphere=realized_atm,
        notes=notes,
    )


def apply_force_models(target: Any, bundle: BuiltForceModelBundle) -> None:
    for model in bundle.models:
        target.addForceModel(model)