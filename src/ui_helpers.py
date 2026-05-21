from __future__ import annotations

from math import isfinite
from typing import Any


def coerce_distance(value: Any) -> float | None:
    if value is None:
        return None
    try:
        distance = float(value)
    except (TypeError, ValueError):
        return None
    if not isfinite(distance):
        return None
    return distance


def format_distance(value: Any) -> str:
    distance = coerce_distance(value)
    if distance is None:
        return "n/a"
    return f"{distance:.3f}"


def average_relevance(items: list[dict[str, Any]]) -> float | None:
    valid_distances = []
    for item in items:
        distance = coerce_distance(item.get("distance"))
        if distance is not None:
            valid_distances.append(distance)

    if not valid_distances:
        return None

    return sum(1 - distance for distance in valid_distances) / len(valid_distances)