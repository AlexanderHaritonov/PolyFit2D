import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Optional

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

    def clone(self) -> 'SequenceSegment':
        """Create a copy of this segment (whole_sequence is shared, line_segment_params is immutable)"""
        return SequenceSegment(
            whole_sequence=self.whole_sequence,  # shared reference, not copied
            first_index=self.first_index,
            last_index=self.last_index,
            line_segment_params=self.line_segment_params  # immutable, safe to share
        )

def plot_segments(segments: list[SequenceSegment]) -> None:
    """Display only the fitted line segments without bitmap or contour"""
    plt.figure(figsize=(10, 8))

    # Draw each segment
    colors = plt.cm.rainbow(np.linspace(0, 1, len(segments)))
    for i, segment in enumerate(segments):
        # Get the points for this segment
        if segment.first_index <= segment.last_index:
            segment_points = segment.whole_sequence[segment.first_index:segment.last_index + 1]
        else:  # Handle circular/closed case
            segment_points = np.vstack([
                segment.whole_sequence[segment.first_index:],
                segment.whole_sequence[:segment.last_index + 1]
            ])

        # Plot segment points and line
        plt.plot(segment_points[:, 1], segment_points[:, 0], 'o-',
                 color=colors[i], linewidth=2.5, markersize=5,
                 label=f'Seg {i} ({segment.points_count()} pts)')

    plt.title(f'Fitted Line Segments ({len(segments)} segments)',
              fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.axis('equal')
    plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('segments_only.png', dpi=150, bbox_inches='tight')
    print("Segments-only plot saved to 'segments_only.png'")
    plt.show()

