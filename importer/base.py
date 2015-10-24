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