import numpy as np

from src.fit_line_segment import fit_line_segment
from src.sequence_segment import SequenceSegment, LineSegmentParams

class FitterToPointsSequence:
    def __init__(self,
                 points_sequence: np.ndarray,
                 is_closed = False,
                 max_segments_count = 30,
                 max_adjust_iterations = 20,
                 tolerance = 2,
                 verbose = False):

        # If closed contour and last point equals first point, remove the duplicate
        if is_closed and np.array_equal(points_sequence[0], points_sequence[-1]):
            self.whole_sequence = points_sequence[:-1]
        else:
            self.whole_sequence = points_sequence
        self.is_closed = is_closed
        self.max_segments_count = max_segments_count
        self.max_adjust_iterations = max_adjust_iterations
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
            if self.verbose:
                print(f"splitting at {index_of_segment_to_split}")

            segmentation_after_split = self.split_segment(segments, index_of_segment_to_split)

            new_variance = self.adjust_segmentation(segmentation_after_split, index_of_segment_to_split)
            if self.verbose: print(f"variance: {new_variance}")

            if new_variance < self.tolerance:
                if self.verbose: print(f"Breaking up at {len(segmentation_after_split)} because variance is less than tolerance.")
                return segmentation_after_split

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

    def points_count(self, first_index, last_index) -> int:
        if last_index > first_index:
            return last_index - first_index + 1
        else:
            return len(self.whole_sequence) - first_index + last_index + 1 # for closed polygon / circular case

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
            line_segment_params=part2_fit)

        # Clone segments before and after the split point
        cloned_before = [seg.clone() for seg in segments[:segment_to_split_index]]
        cloned_after = [seg.clone() for seg in segments[segment_to_split_index+1:]]
        return cloned_before + [segment1, segment2] + cloned_after

    def adjust_segmentation(self, segments: list[SequenceSegment], start_segment_index:int) -> float:
        variance = 0
        for iterations_count in range(self.max_adjust_iterations):
            # The start segment is typically the 1st segment of the pair resulting from the split.
            # Then we do iterations of following (at most MAX_ITERATIONS)
            # we go in the direction of increasing index and for each segment:
            #   1. find out where the border to the next consecutive segment should be
            #   2. adjust where the previous segment ends and the next segment starts
            #   3. re-fit the current segment. Probably can skip re-fitting the next segment as it will be done in the next step anyway
            # when we start again at start_segment_index - 1 and do a pass in the direction of decreasing index,
            # doing steps 1.2.3. for each segment

            def find_adjust_refit(previous_segment, next_segment)->int:
                optimal_last_index = self.best_consecutive_segments_separation(previous_segment, next_segment)
                if optimal_last_index > previous_segment.first_index:
                    count_of_points_changing_segment = abs(optimal_last_index - previous_segment.last_index)
                else: # this might happen with closed polygons
                    old_points_count = previous_segment.points_count()
                    new_points_count = self.points_count(previous_segment.first_index, optimal_last_index)
                    count_of_points_changing_segment = abs(new_points_count - old_points_count)

                if count_of_points_changing_segment > 0:
                    previous_segment.last_index = optimal_last_index
                    next_segment.first_index = (optimal_last_index + 1) % len(self.whole_sequence)

                previous_segment.line_segment_params = fit_line_segment(
                    self.subsequence(previous_segment.first_index, previous_segment.last_index))
                # TODO: consider doing
                # next_segment.line_segment_params = fit_line_segment(self.subsequence(next_segment.first_index, next_segment.last_index))

                return count_of_points_changing_segment

            changes_count = 0
            for direction in [1, -1]:
                start = start_segment_index if direction == 1 else start_segment_index - 1
                if start == -1:
                    start = len(segments) - 2
                stop = len(segments) - 2 if direction == 1 else -1
                #if self.verbose: print(f"adjust_segmentation looping: len:{len(segments)}, start: {start}, stop: {stop}, direction: {direction}")
                for i in range(start, stop, direction):
                    previous_segment, next_segment = segments[i], segments[i+1]
                    count_of_points_changing_segment = find_adjust_refit(previous_segment, next_segment)
                    if count_of_points_changing_segment > 1:
                        changes_count += 1

                if self.is_closed and direction == 1:
                    count_of_points_changing_segment = find_adjust_refit(segments[-1], segments[0])
                    if count_of_points_changing_segment > 1:
                        changes_count += 1

            variance = sum(s.line_segment_params.loss * s.points_count() for s in segments) / len(self.whole_sequence)
            if variance < self.tolerance:
                if self.verbose:
                    print(f"at {len(segments)} segments, {iterations_count} iterations variance smaller than tolerance. Breaking up")
                return variance

            if changes_count == 0:
                if self.verbose:
                    print(f"at {len(segments)} segments, {iterations_count} iterations were no changes. Breaking up")
                return variance

        if self.verbose:
            print(f"at {len(segments)} segments, adjust_segmentation reached {self.max_adjust_iterations} iterations.")
        return variance


    ''' :returns index where segment1 should end'''
    def best_consecutive_segments_separation(self, segment1: SequenceSegment, segment2: SequenceSegment) -> int:
        assert (segment1.last_index == segment2.first_index - 1) or (segment2.first_index==0 and segment1.last_index == len(self.whole_sequence) - 1)
        left_limit = self.get_middle_index(segment1.first_index, segment1.last_index)
        right_limit = self.get_middle_index(segment2.first_index, segment2.last_index)
        relevant_points = self.subsequence(left_limit, right_limit)
        assert relevant_points is not None and relevant_points.shape[0] >= 2
        squared_errors_seg1 = segment1.line_segment_params.squared_distances_to_line(relevant_points)
        squared_errors_seg2 = segment2.line_segment_params.squared_distances_to_line(relevant_points)
        errors_cumsum1 = squared_errors_seg1.cumsum()
        errors_cumsum2 = squared_errors_seg2[::-1].cumsum()
        compound_error_sums = errors_cumsum1[:-1] + errors_cumsum2[-2::-1]
        optimal_last_index = np.argmin(compound_error_sums) # counting from left_limit
        return (int(optimal_last_index) + left_limit) % len(self.whole_sequence)











