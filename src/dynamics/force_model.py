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

    # estimation policy
    estimate_cd: bool = False
    estimate_cr: bool = False
    cd_min: float = 0.5
    cd_max: float = 5.0
    cr_min: float = 0.5
    cr_max: float = 3.0

    # gravity / third body
    gravity_degree: int = 20
    gravity_order: int = 20
    use_third_body: bool = True

    # drag
    drag_enabled: bool = True
    atmosphere: str = "SIMPLE_EXP"
    space_weather_source: str = "AUTO"
    msafe_strength: str = "AVERAGE"
    rho0: float = 3.614e-13
    h0_m: float = 500000.0
    h_scale_m: float = 60000.0

    # J71 external drivers
    j71_f107_avg: float = 150.0
    j71_f107_daily: float = 150.0
    j71_kp_3h: float = 2.0
    j71_base_density_mode: str = "SMOKE_EXP"

    # J71 driver chain
    j71_driver_mode: str = "FIXED"   # FIXED | CSV
    j71_space_weather_csv: Optional[str] = None
    j71_driver_max_age_days: float = 2.0

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


def _resolve_msafe_strength(name: Optional[str]):
    from org.orekit.models.earth.atmosphere.data import MarshallSolarActivityFutureEstimation

    key = str(name or "AVERAGE").strip().upper()

    if key == "WEAK":
        return MarshallSolarActivityFutureEstimation.StrengthLevel.WEAK
    if key == "AVERAGE":
        return MarshallSolarActivityFutureEstimation.StrengthLevel.AVERAGE
    if key == "STRONG":
        return MarshallSolarActivityFutureEstimation.StrengthLevel.STRONG

    raise ValueError(f"Unsupported msafe_strength: {name}")


def _build_jb2008_provider(*, mgr: Any, utc: Any):
    from org.orekit.models.earth.atmosphere.data import JB2008SpaceEnvironmentData

    return JB2008SpaceEnvironmentData(
        JB2008SpaceEnvironmentData.DEFAULT_SUPPORTED_NAMES_SOLFSMY,
        JB2008SpaceEnvironmentData.DEFAULT_SUPPORTED_NAMES_DTC,
        mgr,
        utc,
    )


def _build_atmosphere(*, earth: Any, sun: Any, forces: ForceModelCfg):
    from org.orekit.data import DataContext
    from org.orekit.models.earth.atmosphere import (
        SimpleExponentialAtmosphere, 
        DTM2000, 
        NRLMSISE00, 
        JB2008,
    )
    from org.orekit.models.earth.atmosphere.data import (
        CssiSpaceWeatherData,
        MarshallSolarActivityFutureEstimation,
    )
    from org.orekit.time import TimeScalesFactory

    requested = str(forces.atmosphere or "SIMPLE_EXP").strip().upper()
    notes: List[str] = []

    utc = TimeScalesFactory.getUTC()
    mgr = DataContext.getDefault().getDataProvidersManager()

    def _simple():
        return (
            SimpleExponentialAtmosphere(
                earth,
                float(forces.rho0),
                float(forces.h0_m),
                float(forces.h_scale_m),
            ),
            "SIMPLE_EXP",
        )

    def _cssi():
        return CssiSpaceWeatherData(
            CssiSpaceWeatherData.DEFAULT_SUPPORTED_NAMES,
            mgr,
            utc,
        )

    def _msafe():
        return MarshallSolarActivityFutureEstimation(
            MarshallSolarActivityFutureEstimation.DEFAULT_SUPPORTED_NAMES,
            _resolve_msafe_strength(forces.msafe_strength),
            mgr,
            utc,
        )

    # JACCHIA71 (custom)
    if requested in ("JACCHIA71", "J71", "ATMDEN71"):
        try:
            from src.dynamics.jacchia71 import build_jacchia71_atmosphere

            atmosphere = build_jacchia71_atmosphere(
                earth=earth,
                sun=sun,
                forces=forces,
            )
            notes.append(
                "J71 provisional inputs: "
                f"F107_avg={float(forces.j71_f107_avg)}, "
                f"F107_daily={float(forces.j71_f107_daily)}, "
                f"Kp3h={float(forces.j71_kp_3h)}, "
                f"base_density_mode={str(forces.j71_base_density_mode)}, "
                f"driver_mode={str(forces.j71_driver_mode)}, "
                f"space_weather_csv={str(forces.j71_space_weather_csv)}"
            )
            return atmosphere, "JACCHIA71_CUSTOM", notes
        except Exception as exc:
            raise RuntimeError(f"JACCHIA71 requested but unavailable: {exc}") from exc


    # AUTO / NRLMSISE00
    if requested in ("AUTO", "NRLMSISE00", "NRLMSISE00_CSSI", "NRLMSISE00_MSAFE"):
        if requested in ("AUTO", "NRLMSISE00", "NRLMSISE00_CSSI"):
            try:
                provider = _cssi()
                return NRLMSISE00(provider, sun, earth, utc), "NRLMSISE00_CSSI", notes
            except Exception as exc:
                notes.append(f"NRLMSISE00_CSSI unavailable: {exc}")

        if requested in ("AUTO", "NRLMSISE00", "NRLMSISE00_MSAFE"):
            try:
                provider = _msafe()
                return NRLMSISE00(provider, sun, earth, utc), "NRLMSISE00_MSAFE", notes
            except Exception as exc:
                notes.append(f"NRLMSISE00_MSAFE unavailable: {exc}")

        atm, realized = _simple()
        notes.append(f"Atmosphere '{requested}' fell back to SIMPLE_EXP.")
        return atm, realized, notes

    # DTM2000
    if requested in ("DTM2000", "DTM2000_CSSI", "DTM2000_MSAFE"):
        if requested in ("DTM2000", "DTM2000_CSSI"):
            try:
                provider = _cssi()
                return DTM2000(provider, sun, earth, utc), "DTM2000_CSSI", notes
            except Exception as exc:
                notes.append(f"DTM2000_CSSI unavailable: {exc}")

        if requested in ("DTM2000", "DTM2000_MSAFE"):
            try:
                provider = _msafe()
                return DTM2000(provider, sun, earth, utc), "DTM2000_MSAFE", notes
            except Exception as exc:
                notes.append(f"DTM2000_MSAFE unavailable: {exc}")

        atm, realized = _simple()
        notes.append(f"Atmosphere '{requested}' fell back to SIMPLE_EXP.")
        return atm, realized, notes
    
    # JB2008
    if requested in ("JB2008", "JB2008_SPACEENV"):
        try:
            provider = _build_jb2008_provider(mgr=mgr, utc=utc)
            return JB2008(provider, sun, earth, utc), "JB2008_SPACEENV", notes
        except Exception as exc:
            notes.append(f"JB2008 unavailable: {exc}")
            atm, realized = _simple()
            notes.append("Atmosphere 'JB2008' fell back to SIMPLE_EXP.")
            return atm, realized, notes

    # legacy
    if requested == "SIMPLE_EXP":
        atm, realized = _simple()
        return atm, realized, notes

    atm, realized = _simple()
    notes.append(f"Atmosphere '{requested}' is not implemented; falling back to SIMPLE_EXP.")
    return atm, realized, notes


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
        atmosphere, realized_atm, atm_notes = _build_atmosphere(
            earth=earth,
            sun=sun,
            forces=forces,
        )
        notes.extend(atm_notes)
        
        if bool(forces.estimate_cd):
            drag_sensitive = IsotropicDrag(
                float(forces.area_m2),
                float(forces.cd0),
                float(forces.cd_min),
                float(forces.cd_max),
            )
            notes.append(
                f"Cd solve-for enabled: apriori={float(forces.cd0)}, "
                f"bounds=({float(forces.cd_min)}, {float(forces.cd_max)})"
            )
        else:
            drag_sensitive = IsotropicDrag(
                float(forces.area_m2),
                float(forces.cd0),
            )

        models.append(DragForce(atmosphere, drag_sensitive))
        notes.append(f"Drag atmosphere realized as {realized_atm}")

    # -------------------------
    # SRP
    # -------------------------
    radiation_sensitive = None
    if bool(forces.use_srp):
        srp_area = (
            float(forces.srp_area_m2)
            if forces.srp_area_m2 is not None
            else float(forces.area_m2)
        )

        if bool(forces.estimate_cr):
            radiation_sensitive = IsotropicRadiationSingleCoefficient(
                srp_area,
                float(forces.cr0),
                float(forces.cr_min),
                float(forces.cr_max),
            )
            notes.append(
                f"Cr solve-for enabled: apriori={float(forces.cr0)}, "
                f"bounds=({float(forces.cr_min)}, {float(forces.cr_max)})"
            )
        else:
            radiation_sensitive = IsotropicRadiationSingleCoefficient(
                srp_area,
                float(forces.cr0),
            )

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