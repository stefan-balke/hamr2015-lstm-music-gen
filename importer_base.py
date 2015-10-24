"""Base Class for the dataset importer.

"""

class ImporterBase:
    """Base Class for the dataset import.
    """
    def __init__(self):
        pass

    def import(self):
        raise Exception('Import method must be implemented!')