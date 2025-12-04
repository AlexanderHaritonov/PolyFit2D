import numpy as np

from src.fit_line_segment import fit_line_segment
from src.sequence_segment import SequenceSegment, LineSegmentParams

def fit_to_points_sequence(points_sequence: np.ndarray,
                           is_closed = False,
                           max_segments_count = 30,
                           tolerance = 2,
                           verbose = False) -> list[SequenceSegment]:
    segment_params: LineSegmentParams = fit_line_segment(points_sequence)
    initial_segment = SequenceSegment(
        whole_sequence=points_sequence,
        first_index=0,
        last_index=len(points_sequence)-1,
        line_segment_params=segment_params)
    segments = [initial_segment]

    variance = segment_params.loss
    if variance < tolerance:
        return segments

    while True:
        if len(segments) >= max_segments_count:
            if verbose: print("Max segments count reached. Exiting.")
            return segments

        #variance = sum(s.line_segment_params.loss for s in segments) / len(segments)

        index_of_segment_to_split = choose_segment_index_for_split(segments)
        if index_of_segment_to_split is None:
            if verbose: print("No segments to split. Breaking up at", len(segments), "segments.")
            return segments

        segmentation_after_split = split_segment(segments, index_of_segment_to_split)

        new_variance = adjust_segmentation(segmentation_after_split, tolerance)

        if new_variance > variance - tolerance:
            if verbose: print("Breaking up because of no improvement at", len(segments), "segments.")
            return segments

        segments = segmentation_after_split
        variance = new_variance

def choose_segment_index_for_split(segments: list[SequenceSegment]) -> int | None:
    point_counts = [ segments[i].points_count() for i in range(len(segments)) ]
    eligible = [i for i in range(len(segments)) if point_counts[i] > 3]
    if eligible:
        return max(eligible, key=lambda i: segments[i].line_segment_params.loss / point_counts[i])
    else:
        return None

def split_segment(segments: list[SequenceSegment], index) -> list[SequenceSegment]:
    #TODO
    pass

def adjust_segmentation(points, segments, tolerance=2) -> float:
    #TODO
    pass










