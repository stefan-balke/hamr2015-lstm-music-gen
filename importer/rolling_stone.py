"""Importer for the Rolling Stone 500 Dataset.
   http://theory.esm.rochester.edu/rock_corpus/

"""
import numpy as np
import os
import glob
from base import ImporterBase
import settings


class ImporterRollingStone(ImporterBase):
    """Base Class for the dataset import.
    """
    def __init__(self, beats_per_measure, melody_range, harmony_range, continuation_range, metric_range, path='../data/rock_corpus_v2-1/rs200_melody_nlt'):
        super(ImporterRollingStone, self).__init__(beats_per_measure, melody_range, harmony_range, continuation_range, metric_range)
        self.path = path
        self.output = []
        self.melody_range = melody_range
        self.harmony_range = harmony_range
        self.continuation_range = continuation_range
        self.metric_range = metric_range

        #'pr' stands for piano roll
        self.pr_n_pitches = melody_range[1] - melody_range[0]
        self.pr_bar_division = beats_per_measure

        path_songs = os.path.join(self.path, '*.nlt')

        for cur_path_song in glob.glob(path_songs):
            cur_melody_events = []

            # prevent from trying to read empty files
            if os.stat(cur_path_song).st_size == 0:
                continue

            with open(cur_path_song) as cur_song:
                for cur_event in cur_song:
                    # ignore Error lines
                    if 'Error' in cur_event:
                        continue

                    # every event consists of
                    # Onset, Position in bar, Pitch, Scale Degree
                    cur_event_list = [float(x) for x in cur_event.rstrip().split("\t")]
                    cur_melody_events.append(cur_event_list)

            self.import_piano_roll(np.asarray(cur_melody_events))

    def import_piano_roll(self, note_events):
        # get the number of total bars
        n_bars = int(note_events[-1][1])+1

        # reserve memory
        piano_roll = np.zeros([self.pr_n_pitches, self.pr_bar_division*n_bars])

        pitch_range_start = np.min(note_events[:, 2])
        pitch_range_end = np.max(note_events[:, 2])

        print(pitch_range_start, pitch_range_end, pitch_range_end-pitch_range_start)

        # set the note length as the interval between consecutive metric onsets
        # (suggested in the documentation)
        note_events = np.c_[note_events[:-1], np.diff(note_events[:, 1])]

        # loop over bars
        beat_grid = np.linspace(0, 1, self.pr_bar_division + 1)[:-1]
        np.set_printoptions(precision=3)
        #print(beat_grid)
        for cur_bar in range(int(note_events[0][1]), int(note_events[-1][1])):
            # get the notes which belong to the bar
            cur_notes = note_events[(note_events[:, 1] - cur_bar < 1) & (note_events[:, 1] - cur_bar >= 0), :]

            # what are the nearest notes
            prev_note_idx_end = -1
            for cur_note in cur_notes:
                metric_timing = cur_note[1] - int(cur_note[1])
                # find the closest beat on the beat_grid
                note_idx_start = np.argmin(abs(metric_timing-beat_grid))
                cur_metric_array = self.get_metric_array_from_num_divisions(note_idx_start, self.pr_bar_division)
                print('metric array for note: (' + str(metric_timing) + ', ' + str(note_idx_start) + ') is: ' + str(cur_metric_array))
                note_start_diff = (metric_timing - beat_grid)[note_idx_start]
                duration = int((cur_note[4]+0.01)/ (1.0 / self.pr_bar_division))-1  # round
                note_idx_end = note_idx_start + duration
                cur_pitch = cur_note[2]
                #print('curPitch before key-justification and modding: ' + str(cur_pitch))
                cur_pitch_class = cur_note[3]
                #we want to key-justify our absolute pitch
                pitch_class_diff = cur_pitch % 12 - cur_pitch_class
                if pitch_class_diff < 0:
                    pitch_class_diff += 12
                lowest_octave = int((pitch_range_start - pitch_class_diff) / 12) * 12
                cur_pitch = (cur_pitch - pitch_class_diff - lowest_octave) % 36
               # print('curPitch after key-justification and modding: ' + str(cur_pitch))
                prev_note_idx_end = note_idx_end
                #print(lowest_octave)
                #print(cur_pitch)

                # add to piano-roll
                cur_bar_idx_start = cur_bar*self.pr_bar_division
                cur_bar_idx_end = (cur_bar+1)*self.pr_bar_division
                piano_roll[cur_pitch, cur_bar_idx_start+note_idx_start:cur_bar_idx_start+note_idx_end] = 1
        prev_note_idx_end = -1
        import matplotlib.pyplot as plt
        plt.imshow(piano_roll, cmap=plt.get_cmap('gray_r'))
        plt.show()



        # append to output list
        self.output.append(piano_roll)

    def add_beat_flags(self, off_beat_size, measure_size):
        pass

if __name__ == '__main__':
    data_rs = ImporterRollingStone(settings.BEATS_PER_MEASURE, settings.MELODY_INDICES_RANGE, settings.HARMONY_INDICES_RANGE, settings.CONTINUATION_FLAG_RANGE, settings.METRIC_FLAGS_RANGE)
