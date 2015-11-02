"""Importer for the Rolling Stone 500 Dataset.
   http://theory.esm.rochester.edu/rock_corpus/

"""
import numpy as np
import os
import glob
from base import ImporterBase
import settings
import itertools as it


class ImporterRollingStone(ImporterBase):
    """Base Class for the dataset import.
    """
    def __init__(self, beats_per_measure, melody_range, harmony_range, continuation_range, metric_range,
                 path='data/rock_corpus_v2-1/rs200_melody_nlt',
                 harmony_path='data/rock_corpus_v2-1/rs200_harmony_clt'):
        super(ImporterRollingStone, self).__init__(beats_per_measure, melody_range, harmony_range, continuation_range, metric_range)
        self.path = path
        self.harmony_path = harmony_path
        self.output = []

        #'pr' stands for piano roll
        self.pr_n_pitches = melody_range[1] - melody_range[0]
        self.pr_width = self.metric_range[1]
        self.pr_bar_division = beats_per_measure

        path_songs = os.path.join(self.path, '*.nlt')

        for cur_path_song in glob.glob(path_songs):
            cur_melody_events = []
            cur_chord_events = []

            # prevent from trying to read empty files
            if os.stat(cur_path_song).st_size == 0:
                continue

            print cur_path_song
            cur_path_basename = os.path.basename(cur_path_song)
            harmony_files = os.path.join(self.harmony_path, cur_path_basename[:-7]+ '*')
            harmony_annotations = glob.glob(harmony_files)
            with open(harmony_annotations[0]) as cur_harmony:
                for cur_event in cur_harmony:
                    # ignore Error lines
                    if 'Error' in cur_event:
                        continue
                    all_events = cur_event.rstrip().split("\t")

                    cur_chord_list = []
                    for idx, val in enumerate(all_events):
                        if idx != 2:
                            cur_chord_list.append(float(val))
                        else:
                            cur_chord_list.append(val)

                    #print type(cur_chord_list)
                    cur_chord_events.append(cur_chord_list)

            with open(cur_path_song) as cur_song:
                for cur_event in cur_song:
                    # ignore Error lines
                    if 'Error' in cur_event:
                        continue

                    # every event consists of
                    # Onset, Position in bar, Pitch, Scale Degree
                    cur_event_list = [float(x) for x in cur_event.rstrip().split("\t")]
                    cur_melody_events.append(cur_event_list)

                self.import_piano_roll(np.asarray(cur_melody_events), np.asarray(cur_chord_events))

    def import_piano_roll(self, note_events, chord_events):
        # get the number of total bars
        n_bars = int(note_events[-1][1])+1

        # reserve memory
        piano_roll = np.zeros([self.pr_width, self.pr_bar_division*n_bars])

        pitch_range_start = np.min(note_events[:, 2])
        pitch_range_end = np.max(note_events[:, 2])

        # print(pitch_range_start, pitch_range_end, pitch_range_end-pitch_range_start)

        # set the note length as the interval between consecutive metric onsets
        # (suggested in the documentation)
        note_events = np.c_[note_events[:-1], np.diff(note_events[:, 1])]

        # loop over bars
        beat_grid = np.linspace(0, 1, self.pr_bar_division + 1)[:-1]
        np.set_printoptions(precision=3)
        # print(beat_grid)
        for cur_bar in range(int(note_events[0][1]), int(note_events[-1][1])):
            # get the notes which belong to the bar
            cur_notes = note_events[(note_events[:, 1] - cur_bar < 1) & (note_events[:, 1] - cur_bar >= 0), :]

            # what are the nearest notes
            prev_note_idx_end = -1
            for cur_note in cur_notes:
                metric_timing = cur_note[1] - int(cur_note[1])
                # find the closest beat on the beat_grid
                note_idx_start = np.argmin(abs(metric_timing-beat_grid))
                cur_metric_level = self.get_metric_level_from_num_divisions(note_idx_start, self.pr_bar_division)
                #print('metric array for note: (' + str(metric_timing) + ', ' + str(note_idx_start) + ') is: ' + str(cur_metric_level))
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
                piano_roll[self.metric_range[0] + cur_metric_level, cur_bar_idx_start+note_idx_start] = 1

        #add the harmony

        for i in range(0, len(chord_events) - 1):
            chord_events[i].append(chord_events[i + 1][1] - chord_events[i][1])

        for cur_chord in chord_events:
            if len(cur_chord) < 8:
                break
            cur_chord_bar = int(cur_chord[1])
            chord_bar_idx_start = cur_chord_bar*self.pr_bar_division
            chord_metric_timing = cur_chord[1] - int(cur_chord[1])
            chord_metric_idx = np.argmin(abs(chord_metric_timing-beat_grid))

            chord_duration = int((cur_chord[7]+0.01)/ (1.0 / self.pr_bar_division))-1  # round
            chord_idx_end = chord_metric_idx + chord_duration

            #get the harmony vector
            chord_roman = cur_chord[2]

            upper_vector = [char.isupper() for char in chord_roman]
            is_major = any(upper_vector)

            chord_root_pitch_class = cur_chord[3]
            fifth_pitch_class = (chord_root_pitch_class + 7) % 12

            third_pitch_class = -1
            if is_major == True:
                third_pitch_class = (chord_root_pitch_class + 4) % 12
            else:
                third_pitch_class = (chord_root_pitch_class + 3) % 12

            piano_roll[chord_root_pitch_class + self.harmony_range[0], chord_bar_idx_start+chord_metric_idx:chord_bar_idx_start+chord_idx_end] = 1
            piano_roll[third_pitch_class + self.harmony_range[0], chord_bar_idx_start+chord_metric_idx:chord_bar_idx_start+chord_idx_end] = 1
            piano_roll[fifth_pitch_class + self.harmony_range[0], chord_bar_idx_start+chord_metric_idx:chord_bar_idx_start+chord_idx_end] = 1



        prev_note_idx_end = -1

        #import matplotlib.pyplot as plt
        #plt.imshow(piano_roll, cmap=plt.get_cmap('gray_r'))
        #plt.show()

        # append to output list
        self.output.append(piano_roll)

    def add_beat_flags(self, off_beat_size, measure_size):
        pass

if __name__ == '__main__':
    data_rs = ImporterRollingStone(settings.BEATS_PER_MEASURE, settings.MELODY_INDICES_RANGE, settings.HARMONY_INDICES_RANGE, settings.CONTINUATION_FLAG_RANGE, settings.METRIC_FLAGS_RANGE)
