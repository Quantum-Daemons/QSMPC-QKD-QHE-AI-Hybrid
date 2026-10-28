"""One satellite pass over one station, as the block a downlink's finite key is computed on.

WHY THIS EXISTS (Round 79, the user's choice). The satellite stage reported its key over a fixed
2e7 pulses - a fifth of a second at Micius's 100 MHz - and the key was the asymptotic bound times that
count. The user asked for the block to be sized by link time and for a finite-key length, and asked
for the published practice to decide between seconds at the page's range and a whole pass. Every
flown experiment found computes its finite key over one pass (Micius: Liao et al. 2017 and 2018;
Tiangong-2; Micius to Zvenigorod, Khmelev et al. 2023) or over fixed blocks inside one orbit summed
per orbit (Jinan-1, Li et al. 2025); none computes one on a few seconds of a low-orbit pass. So the
default block is a whole pass, peaking at the range the page sets; pooling several passes (Sidhu et
al., npj Quantum Inf. 8, 18 (2022), Fig. 6; Bourgoin et al., Phys. Rev. A 92, 052339 (2015)) and a
fixed time at one range are offered beside it.

THE GEOMETRY IS `web/src/lib/satNetwork.ts::simulatePass`, step for step, so the two languages
integrate the same pass: a circular orbit; the station a central angle psi0 off the ground track, so
that t seconds from culmination cos psi = cos psi0 cos(omega t); the range from the triangle of radii;
the trapezoid rule at a 1 s step. Earth's rotation is left out (about 0.35 km/s at Xinglong against
the satellite's 7.6), as there.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from .channel import EARTH_RADIUS_KM, ChannelParams, elevation_deg

__all__ = ["GM_EARTH_KM3_S2", "Pass", "central_angle_rad", "orbital_period_s", "range_at_elevation_km",
           "simulate_pass", "top_of_pass_deg"]

#: Earth's gravitational parameter in km^3/s^2 (WGS 84), as `satNetwork.ts` has it.
GM_EARTH_KM3_S2 = 398600.4418


def orbital_period_s(altitude_km: float) -> float:
    """The period of a circular orbit: 94.5 minutes at 500 km."""
    return 2.0 * math.pi * math.sqrt((EARTH_RADIUS_KM + altitude_km) ** 3 / GM_EARTH_KM3_S2)


def central_angle_rad(elevation: float, orbit_km: float, station_km: float = 0.0) -> float:
    """The Earth central angle between the station and the sub-satellite point at `elevation`."""
    a, b, e = EARTH_RADIUS_KM + station_km, EARTH_RADIUS_KM + orbit_km, math.radians(elevation)
    return math.acos(a / b * math.cos(e)) - e


def range_at_elevation_km(elevation: float, orbit_km: float, station_km: float = 0.0) -> float:
    """The slant range at an elevation; the inverse of `channel.elevation_deg`."""
    a, b, s = EARTH_RADIUS_KM + station_km, EARTH_RADIUS_KM + orbit_km, math.sin(math.radians(elevation))
    return math.sqrt(a * a * s * s + b * b - a * a) - a * s


@dataclass(frozen=True)
class Pass:
    """A pass, and the (pulses, channel) segments its finite key is computed on."""

    max_elevation_deg: float
    window_elevation_deg: float
    duration_s: float
    min_range_km: float
    max_range_km: float
    segments: tuple[tuple[float, ChannelParams], ...]

    @property
    def pulses(self) -> float:
        return sum(n for n, _ in self.segments)


def simulate_pass(link: ChannelParams, max_elevation: float, *, window_elevation_deg: float | None = None,
                  step_s: float = 1.0) -> Pass:
    """One pass culminating at `max_elevation` degrees, while the satellite is above the window
    (default: the link's minimum elevation). Empty when the top is below the window."""
    fs = link.free_space
    if fs is None:
        raise ValueError("a pass needs a downlink (ChannelParams.free_space)")
    window = max(fs.min_elevation_deg if window_elevation_deg is None else window_elevation_deg,
                 fs.min_elevation_deg)
    if not (max_elevation >= window) or max_elevation > 90.0:
        return Pass(max_elevation, window, 0.0, math.nan, math.nan, ())
    orbit, station = fs.orbit_altitude_km, fs.station_altitude_km
    a, b = EARTH_RADIUS_KM + station, EARTH_RADIUS_KM + orbit
    w = 2.0 * math.pi / orbital_period_s(orbit)
    psi0 = central_angle_rad(max_elevation, orbit, station)
    psi_w = central_angle_rad(window, orbit, station)
    half = math.acos(min(1.0, math.cos(psi_w) / math.cos(psi0))) / w
    n = max(2, math.ceil(2.0 * half / step_s))
    h = 2.0 * half / n
    # Inside the window by construction; the two ends sit on the window elevation to rounding, so
    # the cut is lifted on each sample rather than letting a last-digit difference drop an end.
    open_sky = replace(fs, min_elevation_deg=0.0)
    segments = []
    for i in range(n + 1):
        t = -half + i * h
        r = math.sqrt(a * a + b * b - 2.0 * a * b * math.cos(psi0) * math.cos(w * t))
        wt = 0.5 * h if i in (0, n) else h
        segments.append((wt * link.source_rate_hz, replace(link, distance_km=r, free_space=open_sky)))
    return Pass(max_elevation, window, 2.0 * half, range_at_elevation_km(max_elevation, orbit, station),
                range_at_elevation_km(window, orbit, station), tuple(segments))


def top_of_pass_deg(link: ChannelParams) -> float | None:
    """The elevation a pass peaking at the link's own range reaches, or None beyond the horizon."""
    fs = link.free_space
    return elevation_deg(link.distance_km, fs.orbit_altitude_km, fs.station_altitude_km)
