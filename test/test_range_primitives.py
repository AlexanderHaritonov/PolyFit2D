"""Pin the shared index-range convention of the range primitives:
first <= last is the linear range [first..last] (equal indices = a single point),
first > last wraps past the end; the full circle is [0..n-1].
"""
import numpy as np

from mask2polymin.sequence_moments import subsequence, range_moments
from mask2polymin.sequence_segment import SequenceSegment
from mask2polymin.fit_to_points_sequence import FitterToPointsSequence

POINTS = np.arange(16, dtype=np.float64).reshape(8, 2)


def test_subsequence_linear_wrap_and_single_point():
    np.testing.assert_array_equal(subsequence(POINTS, 2, 5), POINTS[2:6])
    np.testing.assert_array_equal(subsequence(POINTS, 6, 1), np.vstack([POINTS[6:], POINTS[:2]]))
    np.testing.assert_array_equal(subsequence(POINTS, 3, 3), POINTS[3:4])


def test_points_count_linear_wrap_and_single_point():
    assert SequenceSegment(POINTS, 2, 5).points_count() == 4
    assert SequenceSegment(POINTS, 6, 1).points_count() == 4
    assert SequenceSegment(POINTS, 3, 3).points_count() == 1

    fitter = FitterToPointsSequence(POINTS)
    assert fitter._points_count(2, 5) == 4
    assert fitter._points_count(6, 1) == 4
    assert fitter._points_count(3, 3) == 1


def test_range_moments_count_agrees_with_points_count():
    fitter = FitterToPointsSequence(POINTS)
    for first, last in [(2, 5), (6, 1), (3, 3), (4, 3), (0, 7)]:
        _, count = range_moments(fitter._moments, first, last)
        assert count == fitter._points_count(first, last)


def test_full_circle_is_representable():
    fitter = FitterToPointsSequence(POINTS)
    # both the linear form [0..n-1] and the maximal wrap first == last+1 cover all n points
    assert fitter._points_count(0, 7) == 8
    assert fitter._points_count(4, 3) == 8
    np.testing.assert_array_equal(subsequence(POINTS, 4, 3), np.vstack([POINTS[4:], POINTS[:4]]))
