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

            new_variance = self.adjust_segmentation(segmentation_after_split, index_of_segment_to_split)

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

    ''' computes middle index / pivot point, while respecting the possibility for right_index<left_index because of closed sequence / circularity '''
    def get_middle_index(self, left_index, right_index) -> int:
        if left_index < right_index:
            return (left_index + right_index) // 2
        else:
            mid_index = (left_index + right_index + len(self.whole_sequence)) // 2
            if mid_index < len(self.whole_sequence):
                return mid_index
            else:
                return mid_index - len(self.whole_sequence)

    def subsequence(self, left, right) -> np.ndarray:
        if left < right:
            return self.whole_sequence[left:right+1]
        else:
            return np.vstack([ self.whole_sequence[left:], self.whole_sequence[:right+1] ])

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

    ''' :returns index where segment1 should end'''
    def best_consecutive_segments_separation(self, segment1: SequenceSegment, segment2: SequenceSegment) -> int:
        assert (segment1.last_index == segment2.first_index - 1) or (segment2.first_index==0 and segment1.last_index == len(self.whole_sequence))
        left_limit = self.get_middle_index(segment1.first_index, segment1.last_index)
        right_limit = self.get_middle_index(segment2.first_index, segment2.last_index)
        relevant_points = self.subsequence(left_limit, right_limit)
        assert relevant_points is not None and relevant_points.shape[0] >= 2
        squared_errors_seg1 = segment1.line_segment_params.squared_distances_to_line(relevant_points)
        squared_errors_seg2 = segment2.line_segment_params.squared_distances_to_line(relevant_points)
        errors_cumsum1 = squared_errors_seg1.cumsum()
        errors_cumsum2 = squared_errors_seg2[::-1].cumsum()
        compound_error_sums = errors_cumsum1[:-1] + errors_cumsum2[-2::-1]
        optimal_last_index = np.argmin(compound_error_sums) + 1
        return int(optimal_last_index)











