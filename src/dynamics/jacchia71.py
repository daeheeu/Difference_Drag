from __future__ import annotations

import csv
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from jpype import JImplements, JOverride
from org.hipparchus.geometry.euclidean.threed import Vector3D, FieldVector3D
from org.orekit.utils import PVCoordinates, FieldPVCoordinates

from src.dynamics.j71_table87_3 import (
    J71DensityPolySection,
    DENSITY_POLY_SECTIONS_87_3,
)
from src.dynamics.j71_table87_4 import (
    J71HeliumPolySection,
    HELIUM_POLY_SECTIONS_87_4,
)


@dataclass
class Jacchia71Cfg:
    min_alt_km: float = 90.0
    max_alt_km: float = 2500.0

    # default/fallback fixed drivers
    f107_avg: float = 150.0
    f107_daily: float = 150.0
    kp_3h: float = 2.0

    # provisional base-density mode
    base_density_mode: str = "SMOKE_EXP"
    rho_ref_kg_m3: float = 3.614e-13
    h_ref_km: float = 500.0
    h_scale_km: float = 60.0
    density_scale: float = 1.0

    # optional physics switches
    use_diurnal_variation: bool = False

    # diagnostics
    enable_diagnostics: bool = False
    debug_log_path: Optional[str] = None
    debug_every_n: int = 200
    debug_max_records: int = 2000

    # driver chain
    driver_mode: str = "FIXED"   # FIXED | CSV
    space_weather_csv: Optional[str] = None
    driver_max_age_days: float = 2.0


@dataclass
class Jacchia71Inputs:
    alt_km: float
    lat_rad: float
    lon_rad: float
    day_of_year: Optional[int]
    mjd_utc: Optional[float]
    sun_declination_rad: Optional[float]
    sun_hour_angle_rad: Optional[float]


@dataclass(frozen=True)
class J71SpaceWeatherRow:
    iso_utc: str
    mjd_utc: float
    f107_daily: float
    f107_avg: float
    kp_3h: float


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
        self._space_weather_rows = self._load_space_weather_rows()

        self._debug_call_count = 0
        self._debug_written_count = 0
        self._debug_log_path: Optional[Path] = None

        self._init_debug_log()

    def describe_driver(self) -> str:
        mode = str(self.cfg.driver_mode).strip().upper()

        if mode == "FIXED":
            return (
                "J71 driver FIXED: "
                f"F107_avg={float(self.cfg.f107_avg)}, "
                f"F107_daily={float(self.cfg.f107_daily)}, "
                f"Kp3h={float(self.cfg.kp_3h)}"
            )

        if mode == "CSV":
            if not self._space_weather_rows:
                return (
                    "J71 driver CSV loaded: "
                    f"rows=0, "
                    f"path={str(self.cfg.space_weather_csv)}, "
                    f"max_age_days={float(self.cfg.driver_max_age_days)}"
                )

            first = self._space_weather_rows[0]
            last = self._space_weather_rows[-1]

            return (
                "J71 driver CSV loaded: "
                f"rows={len(self._space_weather_rows)}, "
                f"first={first.iso_utc}, "
                f"last={last.iso_utc}, "
                f"path={str(self.cfg.space_weather_csv)}, "
                f"max_age_days={float(self.cfg.driver_max_age_days)}"
            )

        return f"J71 driver unknown: mode={self.cfg.driver_mode}"

    def _init_debug_log(self) -> None:
        if not bool(self.cfg.enable_diagnostics):
            return

        if not self.cfg.debug_log_path:
            raise RuntimeError(
                "J71 diagnostics enabled but debug_log_path was not provided."
            )

        p = Path(str(self.cfg.debug_log_path))
        p.parent.mkdir(parents=True, exist_ok=True)

        self._debug_log_path = p

        if not p.exists():
            with p.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "call_index",
                    "mjd_utc",
                    "alt_km",
                    "lat_deg",
                    "lon_deg",
                    "sun_decl_deg",
                    "sun_hour_angle_deg",
                    "driver_iso_utc",
                    "f107_avg",
                    "f107_daily",
                    "kp_3h",
                    "t_inf_bar_k",
                    "t_c_k",
                    "t_l_k",
                    "d_t_geomag_k",
                    "texo_k_raw",
                    "texo_k_clamped",
                    "p1",
                    "p2",
                    "p3",
                    "p4",
                    "q1",
                    "q2",
                    "rho_kg_m3",
                ])

    def _maybe_write_debug_row(
        self,
        *,
        j71: Jacchia71Inputs,
        temp_terms: dict[str, float | str | None],
        p1: float,
        p2: float,
        p3: float,
        p4: float,
        q1: float,
        q2: float,
        rho_kg_m3: float,
    ) -> None:
        if not bool(self.cfg.enable_diagnostics):
            return

        self._debug_call_count += 1

        every_n = max(1, int(self.cfg.debug_every_n))
        max_records = max(1, int(self.cfg.debug_max_records))

        if self._debug_written_count >= max_records:
            return

        if (self._debug_call_count - 1) % every_n != 0:
            return

        if self._debug_log_path is None:
            return

        import math

        lat_deg = math.degrees(float(j71.lat_rad))
        lon_deg = math.degrees(float(j71.lon_rad))
        sun_decl_deg = (
            math.degrees(float(j71.sun_declination_rad))
            if j71.sun_declination_rad is not None
            else None
        )
        sun_hour_angle_deg = (
            math.degrees(float(j71.sun_hour_angle_rad))
            if j71.sun_hour_angle_rad is not None
            else None
        )

        with self._debug_log_path.open("a", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                self._debug_call_count,
                j71.mjd_utc,
                j71.alt_km,
                lat_deg,
                lon_deg,
                sun_decl_deg,
                sun_hour_angle_deg,
                temp_terms.get("driver_iso_utc"),
                temp_terms.get("f107_avg"),
                temp_terms.get("f107_daily"),
                temp_terms.get("kp_3h"),
                temp_terms.get("t_inf_bar_k"),
                temp_terms.get("t_c_k"),
                temp_terms.get("t_l_k"),
                temp_terms.get("d_t_geomag_k"),
                temp_terms.get("texo_k_raw"),
                temp_terms.get("texo_k_clamped"),
                p1,
                p2,
                p3,
                p4,
                q1,
                q2,
                rho_kg_m3,
            ])

        self._debug_written_count += 1

    def density_kg_m3(self, date: Any, position: Any, frame: Any) -> float:
        j71 = self._build_inputs(date, position, frame)

        temp_terms = self._exospheric_temperature_terms(j71)
        texo_k = self._clamp_texo_k_to_density_table_range(
            float(temp_terms["texo_k_raw"])
        )
        temp_terms["texo_k_clamped"] = texo_k

        p1 = self._p1_log10_density_g_cm3(j71, texo_k)
        p2 = self._p2_log10_correction(j71)
        p3 = self._p3_log10_correction(j71)
        p4 = self._p4_log10_correction(j71)
        q1 = self._q1_log10_helium_number_density(j71, texo_k)
        q2 = self._q2_log10_helium_correction(j71)

        rho_kg_m3 = self._compose_density_kg_m3(
            p1=p1,
            p2=p2,
            p3=p3,
            p4=p4,
            q1=q1,
            q2=q2,
        )

        self._maybe_write_debug_row(
            j71=j71,
            temp_terms=temp_terms,
            p1=p1,
            p2=p2,
            p3=p3,
            p4=p4,
            q1=q1,
            q2=q2,
            rho_kg_m3=rho_kg_m3,
        )

        return rho_kg_m3

    def _build_inputs(self, date: Any, position: Any, frame: Any) -> Jacchia71Inputs:
        geodetic = self.earth.transform(position, frame, date)

        alt_km = geodetic.getAltitude() / 1000.0
        alt_km = max(self.cfg.min_alt_km, min(self.cfg.max_alt_km, alt_km))

        lat_rad = float(geodetic.getLatitude())
        lon_rad = float(geodetic.getLongitude())

        day_of_year = None
        mjd_utc = None
        sun_declination_rad = None
        sun_hour_angle_rad = None

        try:
            d = date
            if hasattr(date, "toAbsoluteDate"):
                try:
                    d = date.toAbsoluteDate()
                except Exception:
                    d = date

            comp = d.getComponents(self._utc)
            dc = comp.getDate()

            day_of_year = int(dc.getDayOfYear())
            mjd_utc = float(dc.getMJD())
        except Exception:
            day_of_year = None
            mjd_utc = None

        try:
            sun_declination_rad, sun_hour_angle_rad = self._sun_geometry_rad(
                date=date,
                lon_rad=lon_rad,
            )
        except Exception:
            sun_declination_rad = None
            sun_hour_angle_rad = None

        return Jacchia71Inputs(
            alt_km=alt_km,
            lat_rad=lat_rad,
            lon_rad=lon_rad,
            day_of_year=day_of_year,
            mjd_utc=mjd_utc,
            sun_declination_rad=sun_declination_rad,
            sun_hour_angle_rad=sun_hour_angle_rad,
        )
    
    def _sun_geometry_rad(self, *, date: Any, lon_rad: float) -> tuple[float, float]:
        import math

        body_frame = self.earth.getBodyFrame()
        sun_pos = self.sun.getPVCoordinates(date, body_frame).getPosition()

        sx = sun_pos.getX()
        sy = sun_pos.getY()
        sz = sun_pos.getZ()

        if hasattr(sx, "getReal"):
            sx = float(sx.getReal())
            sy = float(sy.getReal())
            sz = float(sz.getReal())
        else:
            sx = float(sx)
            sy = float(sy)
            sz = float(sz)

        r = math.sqrt(sx * sx + sy * sy + sz * sz)
        if r <= 0.0:
            raise ValueError("Sun position norm is zero while computing geometry.")

        # solar declination in body-fixed frame
        decl_rad = math.asin(sz / r)

        # subsolar longitude in body-fixed frame
        sun_lon_rad = math.atan2(sy, sx)

        # local solar hour angle, wrapped to [-pi, +pi]
        hour_angle_rad = lon_rad - sun_lon_rad
        while hour_angle_rad > math.pi:
            hour_angle_rad -= 2.0 * math.pi
        while hour_angle_rad < -math.pi:
            hour_angle_rad += 2.0 * math.pi

        return decl_rad, hour_angle_rad
    

    def _parse_iso_utc(self, s: str) -> dt.datetime:
        s = s.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return dt.datetime.fromisoformat(s).astimezone(dt.timezone.utc)

    def _datetime_to_mjd_utc(self, ts: dt.datetime) -> float:
        unix_s = ts.timestamp()
        jd = unix_s / 86400.0 + 2440587.5
        return jd - 2400000.5

    def _load_space_weather_rows(self) -> list[J71SpaceWeatherRow]:
        mode = str(self.cfg.driver_mode).strip().upper()
        if mode != "CSV":
            return []

        path = self.cfg.space_weather_csv
        if not path:
            raise RuntimeError(
                "J71 driver_mode=CSV but no space_weather_csv path was provided."
            )

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"J71 space weather CSV not found: {p}")

        rows: list[J71SpaceWeatherRow] = []
        with p.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            need = {"iso_utc", "f107_daily", "f107_avg", "kp_3h"}
            if not need.issubset(reader.fieldnames or []):
                raise ValueError(
                    f"J71 space weather CSV must contain columns: {sorted(need)}"
                )

            for row in reader:
                iso_utc = str(row["iso_utc"]).strip()
                ts = self._parse_iso_utc(iso_utc)
                rows.append(
                    J71SpaceWeatherRow(
                        iso_utc=iso_utc,
                        mjd_utc=self._datetime_to_mjd_utc(ts),
                        f107_daily=float(row["f107_daily"]),
                        f107_avg=float(row["f107_avg"]),
                        kp_3h=float(row["kp_3h"]),
                    )
                )

        rows.sort(key=lambda r: r.mjd_utc)
        if not rows:
            raise RuntimeError(f"J71 space weather CSV is empty: {p}")

        return rows

    def _resolve_space_weather_row(self, j71: Jacchia71Inputs) -> J71SpaceWeatherRow:
        mode = str(self.cfg.driver_mode).strip().upper()

        if mode == "FIXED":
            return J71SpaceWeatherRow(
                iso_utc="FIXED",
                mjd_utc=float(j71.mjd_utc or 0.0),
                f107_daily=float(self.cfg.f107_daily),
                f107_avg=float(self.cfg.f107_avg),
                kp_3h=float(self.cfg.kp_3h),
            )

        if mode == "CSV":
            if j71.mjd_utc is None:
                raise RuntimeError("J71 CSV driver requires mjd_utc but it is missing.")

            candidates = [r for r in self._space_weather_rows if r.mjd_utc <= j71.mjd_utc]
            if not candidates:
                raise RuntimeError(
                    f"No J71 space weather row at/before mjd={j71.mjd_utc:.6f}"
                )

            chosen = candidates[-1]
            age_days = float(j71.mjd_utc) - float(chosen.mjd_utc)
            if age_days > float(self.cfg.driver_max_age_days):
                raise RuntimeError(
                    "J71 space weather row is too old: "
                    f"age_days={age_days:.3f}, max_age_days={self.cfg.driver_max_age_days}"
                )

            return chosen

        raise ValueError(f"Unsupported J71 driver_mode: {self.cfg.driver_mode}")    


    def _exospheric_temperature_terms(self, j71: Jacchia71Inputs) -> dict[str, float | str | None]:
        """
        Compute exospheric temperature terms separately so they can be logged
        without changing the force model behavior.
        """
        import math

        sw = self._resolve_space_weather_row(j71)

        f107_avg = float(sw.f107_avg)
        f107_daily = float(sw.f107_daily)
        kp = max(0.0, float(sw.kp_3h))

        # Eq. (8.7-1), (8.7-2)
        t_inf_bar = 379.0 + 3.24 * f107_avg
        t_c = t_inf_bar + 1.3 * (f107_daily - f107_avg)

        # Eq. (8.7-3): optional diurnal variation
        if bool(self.cfg.use_diurnal_variation):
            R = 0.3
            m = 2.2
            n = 3.0
            beta = math.radians(-37.0)
            p = math.radians(6.0)
            gamma = math.radians(43.0)

            delta = j71.sun_declination_rad
            H = j71.sun_hour_angle_rad

            if delta is not None and H is not None:
                delta = float(delta)
                H = float(H)
                phi = float(j71.lat_rad)

                eta = 0.5 * abs(phi - delta)
                theta = 0.5 * abs(phi + delta)

                tau = H + beta + p * math.sin(H + gamma)
                while tau > math.pi:
                    tau -= 2.0 * math.pi
                while tau < -math.pi:
                    tau += 2.0 * math.pi

                a1 = max(0.0, math.sin(theta)) ** m
                a2 = max(0.0, math.cos(eta)) ** m
                a3 = max(0.0, math.cos(tau / 2.0)) ** n

                t_l = t_c * ((1.0 + R * a1) + R * (a2 - a1) * a3)
            else:
                t_l = t_c
        else:
            t_l = t_c

        # Geomagnetic correction
        if j71.alt_km >= 200.0:
            d_t_geomag = 28.0 * kp + 0.03 * math.exp(kp)
        else:
            d_t_geomag = 14.0 * kp + 0.02 * math.exp(kp)

        texo_k = max(183.0, t_l + d_t_geomag)

        return {
            "driver_iso_utc": sw.iso_utc,
            "f107_avg": f107_avg,
            "f107_daily": f107_daily,
            "kp_3h": kp,
            "t_inf_bar_k": t_inf_bar,
            "t_c_k": t_c,
            "t_l_k": t_l,
            "d_t_geomag_k": d_t_geomag,
            "texo_k_raw": texo_k,
            "texo_k_clamped": None,
        }

    def _exospheric_temperature_k(self, j71: Jacchia71Inputs) -> float:
        return float(self._exospheric_temperature_terms(j71)["texo_k_raw"])
    

    def _clamp_texo_k_to_density_table_range(self, texo_k: float) -> float:
        return min(1900.0, max(500.0, float(texo_k)))

    def _p2_log10_correction(self, j71: Jacchia71Inputs) -> float:
        """
        P2 from MicroCosm Eq. (8.7-6), using Eqs. (8.7-7) and (8.7-8).

        Semiannual variation:
            P2 = f(z) * g(t)

        where
            f(z) = [5.876e-7 * z^2.331 + 0.06328] * exp(-2.868e-3 * z)
            g(t) = 0.02835
                 + 0.3817 * [1 + 0.4671 * sin(2*pi*tau + 4.1370)] * sin(4*pi*tau + 4.4259)

            tau = Phi + 0.09544 * ( [0.5 + 0.5*sin(2*pi*Phi + 6.035)]^1.650 - 0.5 )
            Phi = (mjd - 36204.0) / 365.2422
        """
        import math

        if j71.mjd_utc is None:
            return 0.0

        z = float(j71.alt_km)
        mjd = float(j71.mjd_utc)

        phi = (mjd - 36204.0) / 365.2422
        tau = phi + 0.09544 * (
            (0.5 + 0.5 * math.sin(2.0 * math.pi * phi + 6.035)) ** 1.650 - 0.5
        )

        fz = (5.876e-7 * (z ** 2.331) + 0.06328) * math.exp(-2.868e-3 * z)
        gt = 0.02835 + 0.3817 * (
            1.0 + 0.4671 * math.sin(2.0 * math.pi * tau + 4.1370)
        ) * math.sin(4.0 * math.pi * tau + 4.4259)

        return fz * gt

    def _p3_log10_correction(self, j71: Jacchia71Inputs) -> float:
        """
        P3: lower-thermosphere seasonal-latitudinal density correction.

        For the current ~500 km benchmark case this is effectively inactive,
        because MicroCosm documentation states the effect is neglected above
        about 160 km.
        """
        return 0.0

    def _p4_log10_correction(self, j71: Jacchia71Inputs) -> float:
        """
        P4: low-altitude (< 200 km) geomagnetic density correction.

        For the current ~500 km benchmark case this is effectively inactive.
        """
        return 0.0

    def _q1_log10_helium_number_density(
        self,
        j71: Jacchia71Inputs,
        texo_k: float,
    ) -> float:
        """
        Q1 from MicroCosm Table 8.7-4.

        Returns:
            log10(n_He) in number/cm^3

        Note:
        - Table 8.7-4 is only defined from 500 km to 2500 km.
        - Below 500 km, helium contribution is negligible for current purpose.
        """
        if j71.alt_km < 500.0:
            return 0.0

        sections = self._helium_poly_sections_87_4()
        sec = self._select_helium_poly_section(
            sections=sections,
            alt_km=j71.alt_km,
            texo_k=texo_k,
        )
        return self._eval_helium_poly_section(
            sec=sec,
            alt_km=j71.alt_km,
            texo_k=texo_k,
        )

    def _q2_log10_helium_correction(self, j71: Jacchia71Inputs) -> float:
        """
        Q2 from MicroCosm Eq. (8.7-10).

        Helium seasonal-latitudinal correction is:
        - not considered below 500 km
        - neglected for |latitude| < 15 deg in the 500-800 km band
        """
        import math

        if j71.alt_km < 500.0:
            return 0.0

        lat_abs_deg = abs(math.degrees(float(j71.lat_rad)))
        if 500.0 <= j71.alt_km < 800.0 and lat_abs_deg < 15.0:
            return 0.0

        delta = j71.sun_declination_rad
        if delta is None:
            return 0.0

        delta = float(delta)
        abs_delta = abs(delta)
        if abs_delta < 1.0e-12:
            return 0.0

        eps = math.radians(23.44)

        term = math.pi / 4.0 - (float(j71.lat_rad) * delta) / (2.0 * abs_delta)
        ref = math.sin(math.pi / 4.0) ** 3
        seasonal_shape = math.sin(term) ** 3 - ref

        return 0.65 * (abs_delta / eps) * seasonal_shape

    def _p1_log10_density_g_cm3(self, j71: Jacchia71Inputs, texo_k: float) -> float:
        """
        P1 from MicroCosm Table 8.7-3.

        Returns:
            log10(total density) in g/cm^3
        """
        import math

        mode = str(self.cfg.base_density_mode).strip().upper()

        if mode == "SMOKE_EXP":
            rho_kg_m3 = self._base_density_smoke_exp_kg_m3(j71)
            return math.log10(max(rho_kg_m3, 1.0e-30)) - 3.0

        if mode in ("TABLE87_3", "MICROCOSM_TABLE87_3"):
            sections = self._density_poly_sections_87_3()
            sec = self._select_density_poly_section(
                sections=sections,
                alt_km=j71.alt_km,
                texo_k=texo_k,
            )
            return self._eval_density_poly_section(
                sec=sec,
                alt_km=j71.alt_km,
                texo_k=texo_k,
            )

        raise ValueError(
            f"Unsupported J71 base_density_mode: {self.cfg.base_density_mode}"
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

    def _density_poly_sections_87_3(self) -> Sequence[J71DensityPolySection]:
        if not DENSITY_POLY_SECTIONS_87_3:
            raise NotImplementedError(
                "Table 8.7-3 coefficient registry is empty. "
                "Fill src/dynamics/j71_table87_3.py first."
            )
        return DENSITY_POLY_SECTIONS_87_3

    def _select_density_poly_section(
        self,
        *,
        sections: Sequence[J71DensityPolySection],
        alt_km: float,
        texo_k: float,
    ) -> J71DensityPolySection:
        for sec in sections:
            if (
                sec.h_min_km <= float(alt_km) <= sec.h_max_km
                and sec.t_min_k <= float(texo_k) <= sec.t_max_k
            ):
                return sec

        raise ValueError(
            "No J71 density polynomial section matched "
            f"(alt_km={alt_km:.3f}, texo_k={texo_k:.3f})"
        )

    def _eval_density_poly_section(
        self,
        *,
        sec: J71DensityPolySection,
        alt_km: float,
        texo_k: float,
    ) -> float:
        h = float(alt_km)
        t = float(texo_k)

        total = 0.0
        for i, row in enumerate(sec.coeffs):
            h_term = h ** i
            inner = 0.0
            for j, aij in enumerate(row):
                inner += float(aij) * (t ** j)
            total += h_term * inner

        return total

    def _helium_poly_sections_87_4(self) -> Sequence[J71HeliumPolySection]:
        if not HELIUM_POLY_SECTIONS_87_4:
            raise NotImplementedError(
                "Table 8.7-4 coefficient registry is empty. "
                "Fill src/dynamics/j71_table87_4.py first."
            )
        return HELIUM_POLY_SECTIONS_87_4

    def _select_helium_poly_section(
        self,
        *,
        sections: Sequence[J71HeliumPolySection],
        alt_km: float,
        texo_k: float,
    ) -> J71HeliumPolySection:
        for sec in sections:
            if (
                sec.h_min_km <= float(alt_km) <= sec.h_max_km
                and sec.t_min_k <= float(texo_k) <= sec.t_max_k
            ):
                return sec

        raise ValueError(
            "No J71 helium polynomial section matched "
            f"(alt_km={alt_km:.3f}, texo_k={texo_k:.3f})"
        )

    def _eval_helium_poly_section(
        self,
        *,
        sec: J71HeliumPolySection,
        alt_km: float,
        texo_k: float,
    ) -> float:
        """
        Evaluate Q1 = log10(n_He [#/cm^3])
                    = sum_i sum_j b_ij * h^i * T^j
        """
        h = float(alt_km)
        t = float(texo_k)

        total = 0.0
        for i, row in enumerate(sec.coeffs):
            h_term = h ** i
            inner = 0.0
            for j, bij in enumerate(row):
                inner += float(bij) * (t ** j)
            total += h_term * inner

        return total

    def _compose_density_kg_m3(
        self,
        *,
        p1: float,
        p2: float,
        p3: float,
        p4: float,
        q1: float,
        q2: float,
    ) -> float:
        """
        MicroCosm Eq. (8.7-14):

        rho_D = 10^3 * [ 10^(P1 + P2 + P3 + P4) + 10^Q1 * (10^Q2 - 1) * C ]

        where:
        - first bracket term is total density in g/cm^3
        - outer 10^3 converts g/cm^3 -> kg/m^3

        Diagnostic extension in this project:
        - final density is multiplied by cfg.density_scale
        """
        helium_c = 0.6646e-23

        main_term_g_cm3 = 10.0 ** (float(p1) + float(p2) + float(p3) + float(p4))
        helium_term_g_cm3 = (10.0 ** float(q1)) * ((10.0 ** float(q2)) - 1.0) * helium_c

        rho_kg_m3 = 1.0e3 * (main_term_g_cm3 + helium_term_g_cm3)
        rho_kg_m3 *= float(self.cfg.density_scale)

        return max(rho_kg_m3, 1.0e-30) 
    

@JImplements("org.orekit.models.earth.atmosphere.Atmosphere", deferred=True)
class Jacchia71Atmosphere:
    """
    Orekit Atmosphere adapter for Python-side J71 kernel.
    """

    def __init__(self, earth: Any, sun: Any, kernel: Jacchia71DensityKernel):
        self._earth = earth
        self._sun = sun
        self._kernel = kernel

    def get_debug_driver_summary(self) -> str:
        return self._kernel.describe_driver()

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
        density_scale=float(getattr(forces, "j71_density_scale", 1.0)),
        use_diurnal_variation=bool(getattr(forces, "j71_use_diurnal_variation", False)),
        enable_diagnostics=bool(getattr(forces, "j71_enable_diagnostics", False)),
        debug_log_path=getattr(forces, "j71_debug_log_path", None),
        debug_every_n=int(getattr(forces, "j71_debug_every_n", 200)),
        debug_max_records=int(getattr(forces, "j71_debug_max_records", 2000)),
        driver_mode=str(getattr(forces, "j71_driver_mode", "FIXED")),
        space_weather_csv=getattr(forces, "j71_space_weather_csv", None),
        driver_max_age_days=float(getattr(forces, "j71_driver_max_age_days", 2.0)),
    )
    kernel = Jacchia71DensityKernel(earth=earth, sun=sun, cfg=cfg)
    return Jacchia71Atmosphere(earth=earth, sun=sun, kernel=kernel)