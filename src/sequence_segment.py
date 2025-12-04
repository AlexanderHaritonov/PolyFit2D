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

    def points_count(self) -> int:
        if self.last_index > self.first_index:
            return self.last_index - self.first_index + 1
        else:
            return len(self.whole_sequence) - self.first_index + self.last_index + 1 # for closed polygon / circular case

