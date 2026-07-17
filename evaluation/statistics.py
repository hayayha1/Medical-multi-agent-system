from __future__ import annotations

import math
import random
from statistics import mean
from typing import Iterable


def _clean(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def bootstrap_mean_ci(
    values: Iterable[float],
    iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 20260714,
) -> dict[str, float | int]:
    samples = _clean(values)
    if not samples:
        return {"n": 0, "mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = random.Random(seed)
    boot = [mean(rng.choices(samples, k=len(samples))) for _ in range(iterations)]
    alpha = (1 - confidence) / 2
    return {
        "n": len(samples),
        "mean": mean(samples),
        "ci_low": percentile(boot, alpha),
        "ci_high": percentile(boot, 1 - alpha),
    }


def paired_comparison(
    baseline: Iterable[float],
    candidate: Iterable[float],
    iterations: int = 5000,
    seed: int = 20260714,
) -> dict[str, float | int]:
    pairs = [
        (float(left), float(right))
        for left, right in zip(baseline, candidate, strict=True)
        if math.isfinite(float(left)) and math.isfinite(float(right))
    ]
    if not pairs:
        return {"n": 0, "mean_difference": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "p_value": float("nan")}
    differences = [right - left for left, right in pairs]
    rng = random.Random(seed)
    boot = [mean(rng.choices(differences, k=len(differences))) for _ in range(iterations)]
    observed = abs(mean(differences))
    extreme = 0
    for _ in range(iterations):
        randomized = [value if rng.random() < 0.5 else -value for value in differences]
        if abs(mean(randomized)) >= observed:
            extreme += 1
    return {
        "n": len(pairs),
        "mean_difference": mean(differences),
        "ci_low": percentile(boot, 0.025),
        "ci_high": percentile(boot, 0.975),
        "p_value": (extreme + 1) / (iterations + 1),
    }


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [1.0] * len(p_values)
    previous = 1.0
    for reverse_index in range(len(indexed) - 1, -1, -1):
        original_index, value = indexed[reverse_index]
        rank = reverse_index + 1
        corrected = min(previous, value * len(indexed) / rank, 1.0)
        adjusted[original_index] = corrected
        previous = corrected
    return adjusted

