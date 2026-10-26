"""Monotone function inversion via interpolation."""
import numpy as np
from scipy.interpolate import PchipInterpolator
from typing import Optional

class MonotoneInverse:
    """Invert a monotone tabulated function via PCHIP interpolation."""
    
    def __init__(self, x_grid: np.ndarray, y_grid: np.ndarray, decreasing: bool = True):
        """x_grid: domain, y_grid: f(x_grid). decreasing: True if f is decreasing."""
        self.x_grid = np.array(x_grid)
        self.y_grid = np.array(y_grid)
        self.decreasing = decreasing
        assert self.check_monotonicity(), "Input function is not monotone"
        if decreasing:
            # For inversion, swap roles: f^{-1}(y) = x, so interpolate x as function of y
            # y_grid is decreasing, so flip
            self._interp = PchipInterpolator(self.y_grid[::-1], self.x_grid[::-1])
        else:
            self._interp = PchipInterpolator(self.y_grid, self.x_grid)
    
    def check_monotonicity(self) -> bool:
        diffs = np.diff(self.y_grid)
        if self.decreasing:
            return bool(np.all(diffs <= 0))
        else:
            return bool(np.all(diffs >= 0))
    
    def __call__(self, y_val: float) -> float:
        """Return f^{-1}(y_val)."""
        y_min = self.y_grid.min()
        y_max = self.y_grid.max()
        y_clipped = np.clip(y_val, y_min, y_max)
        return float(self._interp(y_clipped))
    
    def __call_array__(self, y_vals: np.ndarray) -> np.ndarray:
        y_min = self.y_grid.min()
        y_max = self.y_grid.max()
        y_clipped = np.clip(y_vals, y_min, y_max)
        return self._interp(y_clipped)
