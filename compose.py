#from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.datasets.data_utils import get_file
import numpy as np
import random
import sys

from importer.rolling_stone import ImporterRollingStone
from importer.essen import ImporterEssen
from importer.essen_untransposed import EssenUntransposed
from exporter.neural_net_to_midi import MidiExporter

from settings import *

# A song is a numpy matrix, giving our boolean features at each 16th note time slice.
SAMPLE_SONG_1 = np.matrix([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,1,1,0,0,0,0],   # C / CMaj
                            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,1],   # C / CMaj
                            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,1,0],   # C / CMaj
                            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,1],   # C / CMaj
                            [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,1,0,0,1,0,0]])  # D / GMaj

SAMPLE_SONG_2 = np.matrix([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,1,1,0,0,0,0],   # C / CMaj
                            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,1],   # C / CMaj
                            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,1,0],   # C / CMaj
                            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,1],   # C / CMaj
                            [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1]])  # E / CMaj

SAMPLE_DATASET = [SAMPLE_SONG_1, SAMPLE_SONG_2]  # Dataset is a list of songs.
SEED = SAMPLE_SONG_1


def sample(a, index_range, temperature=1.0):
    # helper function to sample an index from a probability array
    a = a[index_range[0]:index_range[1]]

    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return index_range[0] + np.argmax(np.random.multinomial(1, a, 1))

class Composer:
    """Base Class for our deep composer.
    """

    window_size = None
    dataset = None  # Format: list of numpy matrices with shape: (time_stamp, features)
    model = None

    def __init__(self, songs=SAMPLE_DATASET, window_size=DEFAULT_WINDOW_SIZE):
        self.dataset = songs
        self.window_size = window_size


    def _compile_model(self):
        # build the model: 2 stacked LSTMs
        print('Building the composer net...')
        print 'Expected vector length:', FEATURE_VECTOR_LENGTH
        num_features = self.dataset[0].shape[0]

        assert (FEATURE_VECTOR_LENGTH == num_features)

        print 'num_features =', num_features
        self.model = Sequential()

        # First LSTM layer
        self.model.add(LSTM(LSTM_HIDDEN_NODES_PER_LAYER, return_sequences=True,
                            input_shape=(self.window_size, num_features)))
        self.model.add(Dropout(DROPOUT_PERCENT))

        # Middle LSTM layers
        for i in range(LSTM_LAYERS-2):
            self.model.add(LSTM(LSTM_HIDDEN_NODES_PER_LAYER, return_sequences=True))
            self.model.add(Dropout(DROPOUT_PERCENT))

        # Final LSTM layer
        self.model.add(LSTM(LSTM_HIDDEN_NODES_PER_LAYER, return_sequences=False))
        self.model.add(Dropout(DROPOUT_PERCENT))

        # Output layer
        self.model.add(Dense(num_features))
        #self.model.add(Activation('relu'))  # Rectified Linear Unit # TODO: 'relu' missing in my theano install
        self.model.add(Activation('sigmoid'))  # Sigmoid

        self.model.compile(loss='mean_squared_error', optimizer='adam')  # adagrad, adadelta, rmsprop


    def train(self, n_epoch=1):
        """Train the neural net
        """

        # Chop up all songs in dataset into examples with window-size N
        training_examples = self._get_training_examples()[0:200]
        if DEBUG:
            training_examples = training_examples[0:DEBUG_NUM_TRAINING_EXAMPLES]
        #print 'training_examples:', training_examples
        print '# training sequences:', len(training_examples)


        # Split into inputs and outputs
        #
        # Input shape: 3D tensor with shape: (nb_samples, timesteps, input_dim).
        # 2D tensor with shape: (nb_samples, output_dim)
        #  (nb_samples, timesteps, input_dim) means:
        #  - nb_samples samples (examples)
        #  - for each sample, a number of time steps (the same for all samples in the batch)
        #  - for each time step of each sample, input_dim features.

        training_examples_X = np.array(tuple(ex[:-1] for ex in training_examples)) # inputs
        training_examples_y = np.array(tuple(np.array(ex[-1:])[0] for ex in training_examples))  # outputs
        #print 'X', training_examples_X
        #print 'y', training_examples_y

        print 'training_examples_X.shape', training_examples_X.shape
        print 'training_examples_y.shape', training_examples_y.shape

        # Build/compile the model
        self._compile_model()

        # Train the model
        for iteration in range(1, 6000):
            print
            print '-' * 50
            print 'Iteration', iteration

            self.model.fit(training_examples_X, training_examples_y, batch_size=BATCH_SIZE, nb_epoch=n_epoch)

            self.compose(iteration)


    def compose(self, index, num_measures=16):
        """Use a pre-trained neural network to compose a melody.
        """
        np.set_printoptions(threshold=np.nan)

        for diversity in [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]:
            print
            print '----- diversity:', diversity

            SEED = self.dataset[3].transpose()
            SEED = SEED[:self.window_size-1]  # Use the window at the start. Subtract 1 since normal window size includes prediction.
            melody = np.expand_dims(SEED, axis=0)
            #print len(SEED)
            #print len(melody)

            for i in range(num_measures * BEATS_PER_MEASURE - len(SEED)):
                #print 'melody.shape', melody.shape
                x = np.expand_dims(np.array(melody[0][i:i + self.window_size]), axis=0)
                #print 'i:', i
                #print 'x:', x
                #print 'x.shape', x.shape

                next_frame = self.model.predict(x, verbose=0)[0]

                #print 'next_frame normalized:', next_frame
                #print 'melody.shape', melody.shape
                #print 'next_frame.shape', next_frame.shape
                print 'next_frame raw:', next_frame

                if SAMPLE_FROM_MELODY_PROBS:
                    # sample from melody probabilities.
                    next_frame = self._sample_melody(next_frame, MELODY_INDICES_RANGE, diversity)
                else:
                    # Winner-takes-all on melody to force monophonic, and force other floats in vector to 0 or 1.
                    next_frame = self._winner_takes_all(next_frame, MELODY_INDICES_RANGE)

                next_frame = self._get_binary_vector(next_frame)
                next_frame = np.expand_dims(next_frame, axis=0)

                #print 'next_frame normalized:', next_frame
                #print 'melody.shape', melody.shape
                #print 'next_frame.shape', next_frame.shape


                melody = np.concatenate([melody, np.expand_dims(next_frame, axis=0)], axis=1)
                #print 'Appended melody:', melody

                # end of for loop

            # Done with melody.
            print 'Final melody:', melody
            print
            # Record the melody matrix to disk.
            melody_csv = 'output/random_%d_%.2f.csv' % (index, diversity)
            np.savetxt(melody_csv, melody[0], fmt='%d', delimiter=',')
            #melody[0].tofile(melody_csv, format='%d', sep=',')
            # Convert to MIDI and write to disk.
            exporter = MidiExporter(melody[0])
            exporter.create_midi_file('output/random_%d_%.2f.midi' % (index, diversity))

    def _get_training_examples(self):
        """Return N - window_size example matrices, each with window_size vectors.
        """
        song_data = []
        for song in self.dataset:
            song = song.transpose()
            song_data.extend(song[i:i+self.window_size] for i in range(0, len(song) - self.window_size + 1))
        return song_data

    def _sample_melody(self, frame_vector, index_range, diversity):
        # Copy vector and zero out the range we care about.
        result = np.array(frame_vector)
        for i in range(index_range[0], index_range[1]):
            result[i] = 0

        # Set the max value in that range to 1.
        result[sample(frame_vector, index_range, diversity)] = 1
        return result

    def _winner_takes_all(self, frame_vector, index_range):
        # Copy vector and zero out the range we care about.
        result = np.array(frame_vector)
        for i in range(index_range[0], index_range[1]):
            result[i] = 0

        # Set the max value in that range to 1.
        result[self._get_max_index_in_range(frame_vector, index_range)] = 1
        return result

    def _get_max_index_in_range(self, frame_vector, index_range):
        max = -1
        argmax = None
        for i in range(index_range[0], index_range[1]):
            if frame_vector[i] > max:
                max = frame_vector[i]
                argmax = i
        return argmax

    def _get_binary_vector(self, frame_vector):
        return np.array([1 if x >= 0.5 else 0 for x in frame_vector])


if __name__ == '__main__':
    data_rs = ImporterRollingStone(BEATS_PER_MEASURE, MELODY_INDICES_RANGE, HARMONY_INDICES_RANGE, CONTINUATION_FLAG_RANGE, METRIC_FLAGS_RANGE)
    #data_essen = EssenUntransposed()

    bach = Composer(data_rs.output)
    #bach = Composer(data_essen.output)
    bach.train()

