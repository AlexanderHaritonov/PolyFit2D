import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Optional

@dataclass
class LineSegmentParams:
    start_point: NDArray[np.float64]
    end_point: NDArray[np.float64]
    direction: NDArray[np.float64]
    loss: float

@dataclass
class SequenceSegment:
    """ whole_sequence: numpy array of shape (N, 2) """
    whole_sequence: np.ndarray
    first_index: int
    last_index: int
    line_segment_params: Optional[LineSegmentParams] = None

