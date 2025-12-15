import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass


@dataclass(frozen=True)
class LineSegmentParams:
    start_point: NDArray[np.float64]
    end_point: NDArray[np.float64]
    direction: NDArray[np.float64]  # unit (length 1) direction vector
    loss: float

    def squared_distances_to_line(self, points: np.ndarray) -> np.ndarray:
        v = points - self.start_point
        # Scalar projection lengths (dot product with direction)
        scalar_proj = np.dot(v, self.direction)  # shape (N,)
        # Vector projections
        proj = np.outer(scalar_proj, self.direction)  # shape (N, D)
        # Perpendicular component
        perp = v - proj
        # Squared distances = squared norm of perpendicular component
        return np.sum(perp ** 2, axis=1)