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

    def add_beat_flags(self):
        raise Exception('add_beat_flags method must be implemented!')