import numpy as np
from src.fit_to_points_sequence import FitterToPointsSequence
from src.sequence_segment import print_segments_info, plot_segments


def test_fit_closed_polygon_four_segments():
    """Test the full fit() method on a closed polygon that should produce 4 segments"""
    points = np.array([
        [0, 0], [0, 1], [0, 2],
        [1, 3], [2, 3], [3, 3],
        [4, 2], [4, 1], [4, 0],
        [3, -1], [2, -1], [1, -1], [0, 0]
    ], dtype=np.float64)

    fitter = FitterToPointsSequence(
        points,
        is_closed=True,
        max_segments_count=10,
        tolerance=0.001,
        verbose=False
    )

    segments = fitter.fit()
    print_segments_info(segments)
    plot_segments(segments)

    # Should result in 4 segments
    assert len(segments) == 4, f"Expected 4 segments, got {len(segments)}"