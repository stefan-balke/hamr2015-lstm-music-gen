"""Base Class for the dataset importer.

"""
import numpy as np


class ImporterBase:
    """Base Class for the dataset import.
    """
    def __init__(self):
        self.output = np.array()
        self.import_piano_roll()
        self.add_beat_flags()

    def import_piano_roll(self):
        raise Exception('import_piano_roll method must be implemented!')

    # off_beat_size - how many 16th notes before the first down beat
    # measure_size - how many 16th notes per measure
    def add_beat_flags(self, off_beat_size, measure_size):
        raise Exception('add_beat_flags method must be implemented!')

    def get_metric_array_from_num_divisions(self, cur_onset_divisions, total_bar_divisions):
        cur_metric_level_divisor = total_bar_divisions
        metric_array = []
        while cur_metric_level_divisor >= 1:
            is_part_of_current_level = False
            if (cur_onset_divisions % cur_metric_level_divisor) == 0:
                is_part_of_current_level = True
            metric_array.append(is_part_of_current_level)
            if is_part_of_current_level == True:
                break
            cur_metric_level_divisor = cur_metric_level_divisor / 2
        return metric_array

