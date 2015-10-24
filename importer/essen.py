"""Importer for the Essen folk song collection
   http://www.esac-data.org/

"""
import numpy as np
from base import ImporterBase


class Essen(ImporterBase):
    """Base Class for the dataset import.
    """
    def import_piano_roll(self):
        pass

    def add_beat_flags(self, off_beat_size, measure_size):
        pass
