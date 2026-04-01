from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import ssl
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen


UTC = dt.timezone.utc
GFZ_API_BASE = "https://kp.gfz.de/app/json/"

def _build_ssl_context(*, insecure: bool) -> ssl.SSLContext:
    if insecure:
        return ssl._create_unverified_context()

    try:
        import certifi  # type: ignore
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _parse_utc(s: str) -> dt.datetime:
    s = s.strip()
    if len(s) == 10:
        # YYYY-MM-DD
        return dt.datetime.fromisoformat(s).replace(tzinfo=UTC)
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    out = dt.datetime.fromisoformat(s)
    if out.tzinfo is None:
        out = out.replace(tzinfo=UTC)
    return out.astimezone(UTC)


def _fmt_utc_z(t: dt.datetime) -> str:
    return t.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _floor_day(t: dt.datetime) -> dt.date:
    return t.astimezone(UTC).date()


def _iter_3h_grid(start: dt.datetime, end: dt.datetime) -> Iterable[dt.datetime]:
    cur = start
    while cur <= end:
        yield cur
        cur += dt.timedelta(hours=3)


def _require_3h_grid_boundary(t: dt.datetime, name: str) -> None:
    t = t.astimezone(UTC)
    if t.minute != 0 or t.second != 0 or t.microsecond != 0:
        raise ValueError(f"{name} must be exactly on an hour boundary: {t.isoformat()}")
    if (t.hour % 3) != 0:
        raise ValueError(f"{name} must lie on a 3-hour grid: {t.isoformat()}")


def _fetch_json(url: str, *, ssl_context: ssl.SSLContext) -> Any:
    req = Request(
        url,
        headers={
            "User-Agent": "Difference_Drag-J71-Builder/1.0",
            "Accept": "application/json",
        },
    )
    with urlopen(req, timeout=60, context=ssl_context) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def _gfz_request(
    *,
    start: dt.datetime,
    end: dt.datetime,
    index: str,
    status: Optional[str] = None,
    ssl_context: ssl.SSLContext,
) -> Tuple[Any, str]:
    params = {
        "start": _fmt_utc_z(start),
        "end": _fmt_utc_z(end),
        "index": index,
    }
    if status:
        params["status"] = status

    url = GFZ_API_BASE + "?" + urlencode(params)
    payload = _fetch_json(url, ssl_context=ssl_context)
    return payload, url


def _pick_first(d: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    for k in keys:
        if k in d:
            return d[k]
    return None


def _parse_float(v: Any) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        return float(v.strip())
    raise TypeError(f"Cannot parse float from value type: {type(v)}")


def _parse_time(v: Any) -> dt.datetime:
    if isinstance(v, dt.datetime):
        return v.astimezone(UTC)
    if isinstance(v, str):
        return _parse_utc(v)
    raise TypeError(f"Cannot parse datetime from value type: {type(v)}")


def _extract_points_from_payload(payload: Any, preferred_value_key: str) -> List[Tuple[dt.datetime, float]]:
    """
    GFZ JSON response shape can vary.
    This parser intentionally supports several common layouts:
      1) [{"time": "...", "value": 3.0}, ...]
      2) [{"datetime": "...", "Kp": 3.0}, ...]
      3) {"data": [...dict items...]}
      4) {"time": [...], "Kp": [...]}
      5) [["2024-05-01T00:00:00Z", 3.0, "def"], ...]
    """
    time_keys = ["time", "datetime", "date", "start", "starttime", "timestamp"]
    value_keys = [
        preferred_value_key,
        preferred_value_key.lower(),
        preferred_value_key.upper(),
        "value",
        "Value",
        "index",
        "Index",
        "kp",
        "Kp",
        "fobs",
        "Fobs",
        "fadj",
        "Fadj",
    ]

    def parse_mapping(obj: Dict[str, Any]) -> List[Tuple[dt.datetime, float]]:
        # case: nested list under "data"
        data = obj.get("data")
        if isinstance(data, list):
            return _extract_points_from_payload(data, preferred_value_key)

        # case: scalar item mapping
        t_raw = _pick_first(obj, time_keys)
        v_raw = _pick_first(obj, value_keys)
        if t_raw is not None and v_raw is not None and not isinstance(t_raw, list) and not isinstance(v_raw, list):
            return [(_parse_time(t_raw), _parse_float(v_raw))]

        # case: array mapping
        t_arr = _pick_first(obj, time_keys)
        v_arr = _pick_first(obj, value_keys)
        if isinstance(t_arr, list) and isinstance(v_arr, list) and len(t_arr) == len(v_arr):
            out: List[Tuple[dt.datetime, float]] = []
            for t_item, v_item in zip(t_arr, v_arr):
                out.append((_parse_time(t_item), _parse_float(v_item)))
            return out

        return []

    # top-level list
    if isinstance(payload, list):
        out: List[Tuple[dt.datetime, float]] = []
        for item in payload:
            if isinstance(item, dict):
                out.extend(parse_mapping(item))
            elif isinstance(item, list) and len(item) >= 2:
                out.append((_parse_time(item[0]), _parse_float(item[1])))
        return out

    # top-level dict
    if isinstance(payload, dict):
        return parse_mapping(payload)

    raise TypeError(f"Unsupported JSON payload type: {type(payload)}")


def _dedup_sort_points(points: List[Tuple[dt.datetime, float]]) -> List[Tuple[dt.datetime, float]]:
    dedup: Dict[dt.datetime, float] = {}
    for t, v in points:
        dedup[t.astimezone(UTC)] = float(v)
    return sorted(dedup.items(), key=lambda x: x[0])


def _build_daily_flux_map(
    *,
    start_day: dt.date,
    end_day: dt.date,
    f107_index: str,
    ssl_context: ssl.SSLContext,
) -> Tuple[Dict[dt.date, float], str]:
    # 81-day centered average needs +/- 40 day buffer
    ext_start = dt.datetime.combine(start_day - dt.timedelta(days=40), dt.time(0, 0, 0), tzinfo=UTC)
    ext_end = dt.datetime.combine(end_day + dt.timedelta(days=40), dt.time(0, 0, 0), tzinfo=UTC)

    payload, url = _gfz_request(
        start=ext_start,
        end=ext_end,
        index=f107_index,
        status=None,
        ssl_context=ssl_context,
    )
    points = _dedup_sort_points(_extract_points_from_payload(payload, f107_index))

    daily: Dict[dt.date, float] = {}
    for t, v in points:
        day = t.astimezone(UTC).date()
        daily[day] = float(v)

    needed_start = start_day - dt.timedelta(days=40)
    needed_end = end_day + dt.timedelta(days=40)

    missing_days: List[str] = []
    cur = needed_start
    while cur <= needed_end:
        if cur not in daily:
            missing_days.append(cur.isoformat())
        cur += dt.timedelta(days=1)

    if missing_days:
        raise RuntimeError(
            "Missing daily F10.7 values required for centered 81-day mean. "
            f"First missing days: {missing_days[:5]}"
        )

    return daily, url


def _centered_mean_81(daily_flux: Dict[dt.date, float], day: dt.date) -> float:
    vals: List[float] = []
    for k in range(-40, 41):
        vals.append(float(daily_flux[day + dt.timedelta(days=k)]))
    return sum(vals) / len(vals)


def _build_kp_map(
    *,
    start: dt.datetime,
    end: dt.datetime,
    kp_status: str,
    ssl_context: ssl.SSLContext,
) -> Tuple[Dict[dt.datetime, float], str]:
    payload, url = _gfz_request(
        start=start,
        end=end,
        index="Kp",
        status=kp_status,
        ssl_context=ssl_context,
    )  

    points = _dedup_sort_points(_extract_points_from_payload(payload, "Kp"))

    kp_map: Dict[dt.datetime, float] = {}
    for t, v in points:
        kp_map[t.astimezone(UTC)] = float(v)

    missing_slots: List[str] = []
    for ts in _iter_3h_grid(start, end):
        if ts not in kp_map:
            missing_slots.append(_fmt_utc_z(ts))

    if missing_slots:
        raise RuntimeError(
            "Missing Kp 3-hour slots from GFZ API response. "
            f"First missing slots: {missing_slots[:5]}"
        )

    return kp_map, url


def _write_csv(
    *,
    out_path: Path,
    start: dt.datetime,
    end: dt.datetime,
    kp_map: Dict[dt.datetime, float],
    daily_flux: Dict[dt.date, float],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["iso_utc", "f107_daily", "f107_avg", "kp_3h"])

        for ts in _iter_3h_grid(start, end):
            day = ts.date()
            f107_daily = float(daily_flux[day])
            f107_avg = float(_centered_mean_81(daily_flux, day))
            kp_3h = float(kp_map[ts])

            w.writerow([
                _fmt_utc_z(ts),
                f"{f107_daily:.6f}",
                f"{f107_avg:.6f}",
                f"{kp_3h:.6f}",
            ])


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build J71 space weather CSV from official GFZ API data."
    )
    ap.add_argument("--start", required=True, help="UTC start, e.g. 2024-05-01T00:00:00Z")
    ap.add_argument("--end", required=True, help="UTC end, e.g. 2024-05-03T00:00:00Z")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument(
        "--f107-index",
        default="Fobs",
        choices=["Fobs", "Fadj"],
        help="GFZ solar flux index to use as daily F10.7 (default: Fobs)",
    )
    ap.add_argument(
        "--kp-status",
        default="def",
        choices=["def", "all"],
        help="GFZ Kp status filter (default: def)",
    )
    ap.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS certificate verification temporarily (dev/debug only)",
    )
    args = ap.parse_args()

    start = _parse_utc(args.start)
    end = _parse_utc(args.end)
    out_path = Path(args.out)

    if end < start:
        raise SystemExit("end must be >= start")

    _require_3h_grid_boundary(start, "start")
    _require_3h_grid_boundary(end, "end")

    start_day = start.date()
    end_day = end.date()

    ssl_context = _build_ssl_context(insecure=bool(args.insecure))

    daily_flux, flux_url = _build_daily_flux_map(
        start_day=start_day,
        end_day=end_day,
        f107_index=str(args.f107_index),
        ssl_context=ssl_context,
    )

    kp_map, kp_url = _build_kp_map(
        start=start,
        end=end,
        kp_status=str(args.kp_status),
        ssl_context=ssl_context,
    )

    _write_csv(
        out_path=out_path,
        start=start,
        end=end,
        kp_map=kp_map,
        daily_flux=daily_flux,
    )

    print("[OK] J71 space weather CSV generated")
    print(f"out       : {out_path}")
    print(f"start     : {_fmt_utc_z(start)}")
    print(f"end       : {_fmt_utc_z(end)}")
    print(f"rows      : {len(list(_iter_3h_grid(start, end)))}")
    print(f"f107 src  : {flux_url}")
    print(f"kp src    : {kp_url}")
    print(f"f107 index: {args.f107_index}")
    print(f"kp status : {args.kp_status}")
    print(f"insecure  : {bool(args.insecure)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())