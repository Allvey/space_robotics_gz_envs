"""Vehicle-scale terrain grades and low-rock traversability, without changing physics."""

import math

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter


def parameters(config):
    settings = dict(
        max_step_height=float(config.get("max_step_height", 0.12)),
        max_slope=float(config.get("max_slope", 0.28)),
        ground_smoothing=float(config.get("ground_smoothing", 0.20)),
        roughness_window=float(config.get("roughness_window", 0.60)),
    )
    if (
        not all(math.isfinite(v) for v in settings.values())
        or settings["max_step_height"] < 0
        or min(settings[k] for k in settings if k != "max_step_height") <= 0
    ):
        raise ValueError("Invalid terrain traversability parameters")
    return settings


def surface(height, resolution, config):
    """Separate sustained ground grade from short, rollable height variations.

    A pebble's sharp edge is not a sustained terrain slope. Test grade on a
    wheel-scale support surface, and compare local residual peak-to-trough
    height with the configured step limit. Unknown cells remain blocked.
    """
    p = parameters(config)
    finite = np.isfinite(height)
    clean = np.where(finite, height, 0)
    sigma = p["ground_smoothing"] / resolution
    weight = gaussian_filter(finite.astype(float), sigma)
    support = gaussian_filter(clean, sigma) / np.maximum(weight, 1e-12)
    dy, dx = np.gradient(support, resolution)
    residual = np.where(finite, height - support, 0)
    width = max(3, 2 * math.ceil(p["roughness_window"] / (2 * resolution)) + 1)
    relief = maximum_filter(residual, size=width) - minimum_filter(residual, size=width)
    if p["max_step_height"] == 0:
        dy, dx = np.gradient(clean, resolution)
        relief_limit = float(config.get("max_height_residual", 0.12))
    else:
        relief_limit = p["max_step_height"]
    blocked = ~finite | (np.hypot(dx, dy) > p["max_slope"]) | (relief > relief_limit)
    return support, blocked


def pool(blocked, subdivisions):
    """Preserve all non-traversable fine evidence covered by a navigation cell."""
    return maximum_filter(blocked, size=2 * math.ceil(subdivisions / 2) + 1)[
        ::subdivisions, ::subdivisions
    ]
