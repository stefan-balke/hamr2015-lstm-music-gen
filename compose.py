from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.datasets.data_utils import get_file
import numpy as np
import random
import sys

BEATS_PER_MEASURE = 16  # 16th note quantization
DEFAULT_WINDOW_SIZE = 3  # number of frames per training window
SAMPLE_DATASET = np.matrix([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,1,1,0,0,0,0],   # C / CMaj
                            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,1],   # C / CMaj
                            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,1,0],   # C / CMaj
                            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,1],   # C / CMaj
                            [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,1,0,0,1,0,0]])  # D / GMaj
SEED = SAMPLE_DATASET

class Composer:
    """Base Class for our deep composer.
    """

    window_size = None
    dataset = None  # each column is one "frame" of data
    model = None

    def __init__(self, window_size=DEFAULT_WINDOW_SIZE):
        self.dataset = self._get_dataset()
        self.window_size = window_size


    def _compile_model(self):
        # build the model: 2 stacked LSTMs
        print('Building the composer net...')
        self.model = Sequential()
        self.model.add(LSTM(50, return_sequences=True, input_shape=(self.window_size, self.dataset.shape[1])))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(50, return_sequences=False))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(self.dataset.shape[1]))
        #self.model.add(Activation('relu'))  # Rectified Linear Unit # TODO: 'relu' missing in my theano install
        self.model.add(Activation('sigmoid'))  # Sigmoid

        self.model.compile(loss='mean_squared_error', optimizer='rmsprop')


    def train(self, n_epoch=1):
        """Train the neural net
        """

        # Chop up dataset into examples with window-size N
        training_examples = self._get_data_subsets()
        print('# training sequences:', len(training_examples))

        # Split into inputs and outputs
        training_examples_X = [ex[:-1] for ex in training_examples]  # inputs
        training_examples_y = [ex[-1:] for ex in training_examples]  # outputs

        # Build/compile the model
        self._compile_model()

        # Train the model
        for iteration in range(1, 60):
            print()
            print('-' * 50)
            print('Iteration', iteration)
            self.model.fit(training_examples_X, training_examples_y, batch_size=128, nb_epoch=n_epoch)

            self.compose()

    def compose(self, num_measures=16):
        """Use a pre-trained neural network to compose a melody.
        """
        melody = [frame for frame in SEED]

        print('----- Generating with seed: "' + melody + '"')

        for i in range(num_measures * BEATS_PER_MEASURE - len(melody_start)):
            x = melody[i:i + self.window_size]
            # TODO: need to add one more dimension with x as only element?

            next_frame = self.model.predict(x, verbose=0)[0]
            melody.append(next_frame)
        print(melody)
        print()

    def _get_dataset(self):
        return SAMPLE_DATASET

    def _get_data_subsets(self):
        """Return N - window_size example matrices, each with window_size vectors.
        """
        return [self.dataset[i:i+self.window_size] for i in range(0, len(self.dataset) - self.window_size)]


if __name__ == '__main__':
    bach = Composer()
    bach.train()
