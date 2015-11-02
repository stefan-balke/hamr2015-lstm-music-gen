__author__ = 'epnichols'

import glob
import os
from exporter.neural_net_to_midi import MidiExporter

import numpy as np
np.set_printoptions(threshold=np.nan)

BASE_PATH = '../output'

if __name__ == '__main__':
    # Loop over all files
    for f in glob.glob('%s/*.csv' % BASE_PATH):
        # Load numpy array.
        melody = np.loadtxt(f, delimiter=',')
        filename = os.path.basename(f)[:-4]
        outfile = '%s/%s.midi' % (BASE_PATH, filename)
        print '\ninput:', f

        exporter = MidiExporter(melody)
        exporter.create_midi_file(outfile)
