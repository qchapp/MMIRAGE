"""Summary-statistics helpers shared by experiment scripts."""

from __future__ import annotations

import statistics
from typing import Any, Iterable, Optional


def mean_std(values: Iterable[Any]) -> dict[str, Optional[float]]:
    clean = [float(value) for value in values if value not in (None, "")]
    if not clean:
        return {"mean": None, "std": None}
    return {
        "mean": round(statistics.mean(clean), 6),
        "std": round(statistics.stdev(clean), 6) if len(clean) > 1 else 0.0,
    }
