"""Yellow–blue colormap matching classic KH / Trixi-style figures."""

from __future__ import annotations

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

KH_CMAP = LinearSegmentedColormap.from_list(
    "kh_yellow_blue",
    [
        (0.02, 0.05, 0.35),
        (0.08, 0.25, 0.65),
        (0.20, 0.55, 0.90),
        (0.75, 0.90, 0.35),
        (1.00, 0.95, 0.15),
    ],
    N=512,
)

# Trixi KH density range
RHO_VMIN = 0.45
RHO_VMAX = 1.95


def upsample_field(field: np.ndarray, factor: int = 2) -> np.ndarray:
    """Bicubic upsample for crisp display (visual only)."""
    if factor <= 1:
        return field
    ny, nx = field.shape
    from scipy.ndimage import zoom

    return zoom(field, factor, order=3).astype(np.float32)
