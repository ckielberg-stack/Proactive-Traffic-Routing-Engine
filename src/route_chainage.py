"""Route-linear chainage helpers for the E4 northbound corridor."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from shapely.geometry import LineString, Point

LatLng = tuple[float, float]

EARTH_RADIUS_M = 6_371_000.0


def build_route_chainage_map(
    positions: Mapping[str | int, LatLng],
    route_points: Sequence[LatLng],
    corridor_length_km: float,
) -> dict[str | int, float]:
    """Project positions onto a route line and return chainage in kilometres."""
    projector = RouteProjector(route_points, corridor_length_km)
    if not projector.is_valid:
        return {}

    chainages: dict[str | int, float] = {}
    for item_id, position in positions.items():
        chainage = projector.project_chainage(position)
        if chainage is not None:
            chainages[item_id] = round(chainage, 2)
    return chainages


def find_nearest_by_chainage(
    target_chainage_km: float,
    candidate_chainages: Mapping[str | int, float],
) -> str | int | None:
    """Return the candidate whose route chainage is closest to the target."""
    if not candidate_chainages:
        return None
    return min(
        candidate_chainages,
        key=lambda candidate_id: abs(candidate_chainages[candidate_id] - target_chainage_km),
    )


class RouteProjector:
    """Project WGS84 lat/lng points onto a route line using local metres."""

    def __init__(
        self,
        route_points: Sequence[LatLng],
        corridor_length_km: float,
    ) -> None:
        self._route_points = [
            (float(lat), float(lng))
            for lat, lng in route_points
            if lat is not None and lng is not None
        ]
        self._corridor_length_km = corridor_length_km
        self._origin = self._route_points[0] if self._route_points else (0.0, 0.0)
        self._ref_lat_rad = math.radians(
            sum(lat for lat, _lng in self._route_points) / len(self._route_points)
        ) if self._route_points else 0.0

        if len(self._route_points) >= 2 and corridor_length_km > 0:
            self._line = LineString([self._to_xy(point) for point in self._route_points])
        else:
            self._line = LineString()

    @property
    def is_valid(self) -> bool:
        return not self._line.is_empty and self._line.length > 0

    def project_chainage(self, position: LatLng) -> float | None:
        """Return route chainage in km for a lat/lng point, or None if invalid."""
        if not self.is_valid:
            return None

        point = Point(self._to_xy(position))
        distance_m = self._line.project(point)
        fraction = distance_m / self._line.length
        return fraction * self._corridor_length_km

    def _to_xy(self, position: LatLng) -> tuple[float, float]:
        lat, lng = position
        origin_lat, origin_lng = self._origin
        x = (
            math.radians(lng - origin_lng)
            * EARTH_RADIUS_M
            * math.cos(self._ref_lat_rad)
        )
        y = math.radians(lat - origin_lat) * EARTH_RADIUS_M
        return x, y
