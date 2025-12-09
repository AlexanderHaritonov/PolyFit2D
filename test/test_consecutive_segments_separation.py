# test_best_consecutive_segments_separation.py

import numpy as np
import pytest
from src.sequence_segment import SequenceSegment, LineSegmentParams
from src.fit_to_points_sequence import FitterToPointsSequence


def test_horizontal_then_vertical():
    points = np.array([
        [0,0], [1,0], [2,0], [3,0],   # horizontal
        [3,1], [3,2], [3,3], [3,4]    # vertical
    ])

    seg1_params = LineSegmentParams(
        start_point=np.array([0,0]),
        end_point=np.array([3,0]),
        direction=np.array([1,0]),
        loss=0.0
    )
    seg1 = SequenceSegment(points, 0, 3, seg1_params)

    seg2_params = LineSegmentParams(
        start_point=np.array([3,0]),
        end_point=np.array([3,4]),
        direction=np.array([0,1]),
        loss=0.0
    )
    seg2 = SequenceSegment(points, 4, 7, seg2_params)

    fitter = FitterToPointsSequence(points)
    idx = fitter.best_consecutive_segments_separation(seg1, seg2)

    # Expect separation near the end of horizontal (index 3)
    assert idx == 3


def test_diagonal_then_horizontal():
    points = np.array([
        [0,0], [1,1], [2,2], [3,3],   # diagonal
        [4,3], [5,3], [6,3], [7,3]    # horizontal
    ])

    seg1_params = LineSegmentParams(
        start_point=np.array([0,0]),
        end_point=np.array([3,3]),
        direction=np.array([1/np.sqrt(2), 1/np.sqrt(2)]),
        loss=0.0
    )
    seg1 = SequenceSegment(points, 0, 3, seg1_params)

    seg2_params = LineSegmentParams(
        start_point=np.array([3,3]),
        end_point=np.array([7,3]),
        direction=np.array([1,0]),
        loss=0.0
    )
    seg2 = SequenceSegment(points, 4, 7, seg2_params)

    fitter = FitterToPointsSequence(points)
    idx = fitter.best_consecutive_segments_separation(seg1, seg2)

    assert idx == 3


def test_noisy_horizontal_then_vertical():
    points = np.array([
        [0,0], [1,0.1], [2,-0.1], [3,0.05],   # noisy horizontal
        [3.1,1], [2.9,2], [3.05,3], [3,4.1]   # noisy vertical
    ])

    seg1_params = LineSegmentParams(
        start_point=np.array([0,0]),
        end_point=np.array([3,0]),
        direction=np.array([1,0]),
        loss=0.1
    )
    seg1 = SequenceSegment(points, 0, 3, seg1_params)

    seg2_params = LineSegmentParams(
        start_point=np.array([3,0]),
        end_point=np.array([3,4]),
        direction=np.array([0,1]),
        loss=0.1
    )
    seg2 = SequenceSegment(points, 4, 7, seg2_params)

    fitter = FitterToPointsSequence(points)
    idx = fitter.best_consecutive_segments_separation(seg1, seg2)

    # With noise, still expect separation near index 3
    assert idx in (2, 3, 4)


def test_closed_polygon_square():
    points = np.array([
        [0,0], [1,0], [2,0], [2,1], [2,2], [1,2], [0,2], [0,1], [0,0]
    ])

    seg1_params = LineSegmentParams(
        start_point=np.array([0,0]),
        end_point=np.array([2,0]),
        direction=np.array([1,0]),
        loss=0.0
    )
    seg1 = SequenceSegment(points, 0, 2, seg1_params)

    seg2_params = LineSegmentParams(
        start_point=np.array([2,0]),
        end_point=np.array([2,2]),
        direction=np.array([0,1]),
        loss=0.0
    )
    seg2 = SequenceSegment(points, 3, 5, seg2_params)

    fitter = FitterToPointsSequence(points, is_closed=True)
    idx = fitter.best_consecutive_segments_separation(seg1, seg2)

    assert idx == 2