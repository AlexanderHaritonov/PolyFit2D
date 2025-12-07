import numpy as np

from src.fit_line_segment import fit_line_segment
from src.sequence_segment import SequenceSegment, LineSegmentParams

class FitterToPointsSequence:
    def __init__(self,
                 points_sequence: np.ndarray,
                 is_closed = False,
                 max_segments_count = 30,
                 max_adjust_iterations = 30,
                 tolerance = 2,
                 verbose = False):
        self.whole_sequence = points_sequence
        self.is_closed = is_closed
        self.max_segments_count = max_segments_count
        self.tolerance = tolerance
        self.verbose = verbose

    def fit_to_points_sequence(self) -> list[SequenceSegment]:
        segment_params: LineSegmentParams = fit_line_segment(self.whole_sequence)
        initial_segment = SequenceSegment(
            whole_sequence=self.whole_sequence,
            first_index=0,
            last_index=len(self.whole_sequence)-1,
            line_segment_params=segment_params)
        segments = [initial_segment]

        variance = segment_params.loss
        if variance < self.tolerance:
            return segments

        while True:
            if len(segments) >= self.max_segments_count:
                if self.verbose: print("Max segments count reached. Exiting.")
                return segments

            #variance = sum(s.line_segment_params.loss for s in segments) / len(segments)

            index_of_segment_to_split = self.choose_segment_index_for_split(segments)
            if index_of_segment_to_split is None:
                if self.verbose: print("No segments to split. Breaking up at", len(segments), "segments.")
                return segments

            segmentation_after_split = self.split_segment(segments, index_of_segment_to_split)

            new_variance = self.adjust_segmentation(segmentation_after_split, self.tolerance)

            if new_variance > variance - self.tolerance:
                if self.verbose: print("Breaking up because of no improvement at", len(segments), "segments.")
                return segments

            segments = segmentation_after_split
            variance = new_variance

    @staticmethod
    def choose_segment_index_for_split(segments: list[SequenceSegment]) -> int | None:
        point_counts = [ segments[i].points_count() for i in range(len(segments)) ]
        eligible = [i for i in range(len(segments)) if point_counts[i] > 3]
        if eligible:
            return max(eligible, key=lambda i: segments[i].line_segment_params.loss / point_counts[i])
        else:
            return None

    def split_segment(self, segments: list[SequenceSegment], segment_to_split_index: int) -> list[SequenceSegment]:
        segment = segments[segment_to_split_index]
        if segment.first_index < segment.last_index:
            pivot_point_index = (segment.first_index + segment.last_index) // 2
            part1 = self.whole_sequence[segment.first_index : pivot_point_index + 1]
            part2 = self.whole_sequence[pivot_point_index + 1 : segment.last_index + 1]
        else: # can happen if is_closed
            pivot_point_index = (segment.first_index + segment.last_index + len(self.whole_sequence)) // 2
            if pivot_point_index < len(self.whole_sequence):
                part1 = self.whole_sequence[segment.first_index : pivot_point_index + 1]
                part2 = np.vstack([self.whole_sequence[pivot_point_index + 1 :], self.whole_sequence[: segment.last_index + 1]])
            else:
                pivot_point_index -= len(self.whole_sequence)
                part1 = np.vstack([self.whole_sequence[segment.first_index:], self.whole_sequence[:pivot_point_index + 1]])
                part2 = self.whole_sequence[pivot_point_index + 1 : segment.last_index + 1]

        part1_fit: LineSegmentParams = fit_line_segment(part1)
        segment1 = SequenceSegment(
            whole_sequence= self.whole_sequence,
            first_index= segment.first_index,
            last_index= pivot_point_index,
            line_segment_params=part1_fit)
        part2_fit: LineSegmentParams = fit_line_segment(part2)
        seg2_first_index = pivot_point_index + 1
        if seg2_first_index >= len(self.whole_sequence):
            seg2_first_index -= len(self.whole_sequence)
        segment2 = SequenceSegment(
            whole_sequence= self.whole_sequence,
            first_index= seg2_first_index,
            last_index= segment.last_index,
            line_segment_params=part1_fit)

        return segments[:segment_to_split_index] + [segment1, segment2] + segments[segment_to_split_index+1:]

    def adjust_segmentation(self, segments, start_segment) -> float:
        #TODO
        pass










