"""Base Class for the dataset importer.

"""
import numpy as np


class ImporterBase(object):
    """Base Class for the dataset import.
    """
    def __init__(self, beats_per_measure, melody_range, harmony_range, continuation_range, metric_range):
        self.melody_range = melody_range
        self.harmony_range = harmony_range
        self.continuation_range = continuation_range
        self.metric_range = metric_range

        #'pr' stands for piano roll
        self.num_pitches = melody_range[1] - melody_range[0]
        self.bar_divisions = beats_per_measure

    def import_piano_roll(self):
        raise Exception('import_piano_roll method must be implemented!')

    # off_beat_size - how many 16th notes before the first down beat
    # measure_size - how many 16th notes per measure
    def add_beat_flags(self, off_beat_size, measure_size):
        raise Exception('add_beat_flags method must be implemented!')

    def get_metric_level_from_num_divisions(self, cur_onset_divisions, total_bar_divisions):
        cur_metric_level_divisor = total_bar_divisions
        metric_array = []
        level = 0
        while cur_metric_level_divisor >= 1:
            if (cur_onset_divisions % cur_metric_level_divisor) == 0:
                break
            level += 1
            cur_metric_level_divisor = cur_metric_level_divisor / 2
        return level

