from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from jpype import JImplements, JOverride
from org.hipparchus.geometry.euclidean.threed import Vector3D, FieldVector3D
from org.orekit.utils import PVCoordinates, FieldPVCoordinates


@dataclass
class Jacchia71Cfg:
    min_alt_km: float = 90.0
    max_alt_km: float = 2500.0

    # provisional external drivers
    f107_avg: float = 150.0
    f107_daily: float = 150.0
    kp_3h: float = 2.0

    # provisional base-density smoke mode
    base_density_mode: str = "SMOKE_EXP"
    rho_ref_kg_m3: float = 3.614e-13
    h_ref_km: float = 500.0
    h_scale_km: float = 60.0


@dataclass
class Jacchia71Inputs:
    alt_km: float
    lat_rad: float
    lon_rad: float
    day_of_year: Optional[int]


class Jacchia71DensityKernel:
    """
    Pure-Python J71 density kernel.

    이번 단계의 목적:
    - 기존의 "한 줄 raise" 구조를
      "입력 구성 -> 외기권 온도 -> 비온도 항 -> 기본 밀도 -> 최종 조합"
      구조로 분해한다.
    - 아직 수식은 다 넣지 않는다.
    - 어느 단계가 미구현인지 명확히 드러나게 만든다.
    """

    def __init__(self, earth: Any, sun: Any, cfg: Optional[Jacchia71Cfg] = None):
        self.earth = earth
        self.sun = sun
        self.cfg = cfg or Jacchia71Cfg()

        from org.orekit.time import TimeScalesFactory

        self._utc = TimeScalesFactory.getUTC()

    def density_kg_m3(self, date: Any, position: Any, frame: Any) -> float:
        j71 = self._build_inputs(date, position, frame)

        texo_k = self._exospheric_temperature_k(j71)
        delta_log10_rho = self._temperature_independent_log10_terms(j71)
        base_log10_rho = self._base_log10_density(j71, texo_k)

        return self._compose_density_kg_m3(
            base_log10_rho=base_log10_rho,
            delta_log10_rho=delta_log10_rho,
        )

    def _build_inputs(self, date: Any, position: Any, frame: Any) -> Jacchia71Inputs:
        geodetic = self.earth.transform(position, frame, date)

        alt_km = geodetic.getAltitude() / 1000.0
        alt_km = max(self.cfg.min_alt_km, min(self.cfg.max_alt_km, alt_km))

        lat_rad = float(geodetic.getLatitude())
        lon_rad = float(geodetic.getLongitude())

        day_of_year = None
        try:
            d = date
            if hasattr(date, "toAbsoluteDate"):
                try:
                    d = date.toAbsoluteDate()
                except Exception:
                    d = date

            comp = d.getComponents(self._utc)
            day_of_year = int(comp.getDate().getDayOfYear())
        except Exception:
            day_of_year = None

        return Jacchia71Inputs(
            alt_km=alt_km,
            lat_rad=lat_rad,
            lon_rad=lon_rad,
            day_of_year=day_of_year,
        )

    def _exospheric_temperature_k(self, j71: Jacchia71Inputs) -> float:
        """
        Provisional J71 exospheric temperature model.

        Implemented now:
        - Eq. (8.7-1): T_inf_bar = 379 + 3.24 * F107_avg
        - Eq. (8.7-2): T_c      = T_inf_bar + 1.3 * (F107_daily - F107_avg)

        Geomagnetic correction:
        - Above 200 km: Eq. (8.7-4)
            dT = 28 * Kp + 0.03 * exp(Kp)
        - Below 200 km: use temperature part of Eq. (8.7-5)(b)
            dT = 14 * Kp + 0.02 * exp(Kp)

        Deferred to later step:
        - Diurnal variation
        - P4 density correction for h < 200 km
        - Tables.dat driven real flux/Kp chain
        """
        import math

        f107_avg = float(self.cfg.f107_avg)
        f107_daily = float(self.cfg.f107_daily)
        kp = max(0.0, float(self.cfg.kp_3h))

        t_inf_bar = 379.0 + 3.24 * f107_avg
        t_c = t_inf_bar + 1.3 * (f107_daily - f107_avg)

        if j71.alt_km >= 200.0:
            d_t_geomag = 28.0 * kp + 0.03 * math.exp(kp)
        else:
            d_t_geomag = 14.0 * kp + 0.02 * math.exp(kp)

        texo_k = t_c + d_t_geomag

        # simple sanity floor
        return max(183.0, texo_k)
    

    def _temperature_independent_log10_terms(self, j71: Jacchia71Inputs) -> float:
        """
        Provisional placeholder for J71 temperature-independent corrections.

        Future terms to be added here:
        - seasonal-latitudinal variation of lower thermosphere
        - seasonal-latitudinal variation of helium
        - semi-annual variation

        For now, return 0.0 so execution can advance to _base_log10_density().
        """
        return 0.0

    def _base_log10_density(self, j71: Jacchia71Inputs, texo_k: float) -> float:
        """
        Provisional smoke-test base density.

        IMPORTANT:
        - This is NOT the real MicroCosm J71 polynomial fit.
        - It only exists to let the custom J71 atmosphere run end-to-end.
        - Do NOT use resulting OD/OP accuracy as J71 parity evidence.

        Real target later:
        - piecewise polynomial fit from MicroCosm Table 8.7-3
        """
        import math

        mode = str(self.cfg.base_density_mode).strip().upper()

        if mode == "SMOKE_EXP":
            rho = self._base_density_smoke_exp_kg_m3(j71)
            return math.log10(max(rho, 1.0e-30))

        raise NotImplementedError(
            "J71 step not implemented yet: real _base_log10_density polynomial "
            f"(mode={mode}, alt_km={j71.alt_km:.3f}, texo_k={texo_k})"
        )
    
    def _base_density_smoke_exp_kg_m3(self, j71: Jacchia71Inputs) -> float:
        """
        Very simple exponential surrogate used only for runtime smoke testing.
        Anchored to rho_ref_kg_m3 at h_ref_km with scale height h_scale_km.
        """
        import math

        dh_km = float(j71.alt_km) - float(self.cfg.h_ref_km)
        rho = float(self.cfg.rho_ref_kg_m3) * math.exp(
            -dh_km / float(self.cfg.h_scale_km)
        )
        return max(rho, 1.0e-30)

    def _compose_density_kg_m3(
        self,
        *,
        base_log10_rho: float,
        delta_log10_rho: float,
    ) -> float:
        log10_rho = float(base_log10_rho) + float(delta_log10_rho)
        return 10.0 ** log10_rho
    

@JImplements("org.orekit.models.earth.atmosphere.Atmosphere", deferred=True)
class Jacchia71Atmosphere:
    """
    Orekit Atmosphere adapter for Python-side J71 kernel.
    """

    def __init__(self, earth: Any, sun: Any, kernel: Jacchia71DensityKernel):
        self._earth = earth
        self._sun = sun
        self._kernel = kernel

    @JOverride
    def getFrame(self):
        return self._earth.getBodyFrame()

    @JOverride
    def getDensity(self, *args):
        """
        Supports both overloads:

        - getDensity(AbsoluteDate, Vector3D, Frame)
        - getDensity(FieldAbsoluteDate<T>, FieldVector3D<T>, Frame)
        """
        if len(args) != 3:
            raise TypeError(f"getDensity expects 3 args, got {len(args)}")

        date, position, frame = args
        rho = float(self._kernel.density_kg_m3(date, position, frame))

        x = position.getX()
        if hasattr(x, "newInstance"):
            return x.newInstance(rho)

        return rho

    @JOverride
    def getVelocity(self, *args):
        """
        Explicitly implement Atmosphere.getVelocity(...) because the Orekit
        default interface method is not being dispatched correctly through
        the Python proxy.

        Supports both overloads:
        - getVelocity(AbsoluteDate, Vector3D, Frame)
        - getVelocity(FieldAbsoluteDate<T>, FieldVector3D<T>, Frame)

        Semantics follow Orekit default behavior:
        atmosphere molecules have zero velocity in the body frame.
        """
        if len(args) != 3:
            raise TypeError(f"getVelocity expects 3 args, got {len(args)}")

        date, position, frame = args
        body_frame = self._earth.getBodyFrame()

        x = position.getX()

        # Field overload
        if hasattr(x, "newInstance"):
            frame_to_body = frame.getTransformTo(body_frame, date)
            p_body = frame_to_body.transformPosition(position)

            zero = FieldVector3D(
                x.newInstance(0.0),
                x.newInstance(0.0),
                x.newInstance(0.0),
            )

            pv_body = FieldPVCoordinates(p_body, zero)

            body_to_frame = body_frame.getTransformTo(frame, date)
            pv_frame = body_to_frame.transformPVCoordinates(pv_body)
            return pv_frame.getVelocity()

        # Regular overload
        frame_to_body = frame.getTransformTo(body_frame, date)
        p_body = frame_to_body.transformPosition(position)

        pv_body = PVCoordinates(p_body, Vector3D.ZERO)

        body_to_frame = body_frame.getTransformTo(frame, date)
        pv_frame = body_to_frame.transformPVCoordinates(pv_body)
        return pv_frame.getVelocity()


def build_jacchia71_atmosphere(*, earth: Any, sun: Any, forces: Any) -> Any:
    cfg = Jacchia71Cfg(
        f107_avg=float(getattr(forces, "j71_f107_avg", 150.0)),
        f107_daily=float(getattr(forces, "j71_f107_daily", 150.0)),
        kp_3h=float(getattr(forces, "j71_kp_3h", 2.0)),
        base_density_mode=str(getattr(forces, "j71_base_density_mode", "SMOKE_EXP")),
        rho_ref_kg_m3=float(getattr(forces, "rho0", 3.614e-13)),
        h_ref_km=float(getattr(forces, "h0_m", 500000.0)) / 1000.0,
        h_scale_km=float(getattr(forces, "h_scale_m", 60000.0)) / 1000.0,
    )
    kernel = Jacchia71DensityKernel(earth=earth, sun=sun, cfg=cfg)
    return Jacchia71Atmosphere(earth=earth, sun=sun, kernel=kernel)