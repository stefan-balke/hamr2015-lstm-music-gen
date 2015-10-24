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

class Composer:
    """Base Class for our deep composer.
    """

    dataset = None  # each column is one "frame" of data
    model = None

    def __init__(self):
        self.dataset = _get_dataset()


    def _compile_model(self, window_size):
        # build the model: 2 stacked LSTMs
        print('Building the composer net...')
        self.model = Sequential()
        self.model.add(LSTM(50, return_sequences=True, input_shape=(window_size, self.dataset.shape[1])))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(50, return_sequences=False))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(self.dataset.shape[1]))
        self.model.add(Activation('relu'))  # Rectified Linear Unit

        self.model.compile(loss='mean_squared_error', optimizer='rmsprop')


    def train(self, n_epoch=1, window_size=DEFAULT_WINDOW_SIZE):
        """Train the neural net
        """

        # Chop up dataset into examples with window-size N
        training_examples = self._get_data_subsets(window_size=window_size)
        print('# training sequences:', len(training_examples))

        # Split into inputs and outputs
        training_examples_X = [ex[:-1] for ex in training_examples]  # inputs
        training_examples_y = [ex[-1:] for ex in training_examples]  # outputs

        # Build/compile the model
        self._compile_model(window_size)

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
        melody_start = SAMPLE_DATASET
        print('----- Generating with seed: "' + melody_start + '"')

        #sys.stdout.write(generated)

        for iteration in range(num_measures * BEATS_PER_MEASURE - len(melody_start)):
            pass
            # preds = model.predict(x, verbose=0)[0]
            # next_index = sample(preds, diversity)
            # next_char = indices_char[next_index]
            #
            # generated += next_char
            # sentence = sentence[1:] + next_char
            #
            # sys.stdout.write(next_char)
            # sys.stdout.flush()
        print()

    def _get_dataset(self):
        return SAMPLE_DATASET

    def _get_data_subsets(self, window_size):
        """Return N - window_size example matrices, each with window_size vectors.
        """
        return (self.dataset[i:i+window_size] for i in range(0, len(self.dataset) - window_size))


if __name__ == '__main__':
    pass
