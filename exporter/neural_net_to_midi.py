__author__ = 'epnichols'

from settings import *

from music21 import stream, note, chord, scale, duration, pitch
import music21.midi as midi

import numpy as np

class MidiExporter(object):
    song = None

    def __init__(self, song_array):
        """song_array: numpy matrix of features for each frame.
        """
        self.song = song_array

        print 'Feature vector length:', FEATURE_VECTOR_LENGTH
        print 'Melody bits:', MELODY_INDICES_RANGE
        print 'Harmony bits:', HARMONY_INDICES_RANGE
        print 'Continuation bits:', CONTINUATION_FLAG_RANGE
        print 'Metric flag bits:', METRIC_FLAGS_RANGE

        #print 'song:', self.song
        print 'song.shape', self.song.shape


    def create_midi_file(self, filename):
        # s = stream.Stream()
        # n = note.Note('g#')
        # n.quarterLength = .5
        # s.repeatAppend(n, 4)
        # mf = midi.translate.streamToMidiFile(s)

        # sc = scale.PhrygianScale('g')
        # x = [s.append(note.Note(sc.pitchFromDegree(i % 11), quarterLength=.25)) for i in range(60)]

        # Melody stream.
        s = stream.Part()
        s.id = 'melody'
        num_frames = 0
        prev_n = -1
        new_note = None
        for idx, frame in enumerate(self.song):
            #print idx
            melody = frame[MELODY_INDICES_RANGE[0]:MELODY_INDICES_RANGE[1]]
            print melody
            n = None
            for i, val in enumerate(melody):
                if val == 1:
                    n = i
                    break

            # If the pitch changes, commit previous note and start new one.
            if n != prev_n:
                prev_n = n
                # commit the previous note.
                if new_note:
                    new_note.quarterLength = .25 * num_frames
                    s.append(new_note)
                # start a new note.
                if n is not None:
                    new_note = note.Note(i + 60)
                else:
                    # rest
                    new_note = note.Rest()
                num_frames = 1
            else:  # TODO: test for continuity flag
                # note is continued. Just add to the duration.
                num_frames += 1

        # Commit the final note
        new_note.quarterLength = .25 * num_frames
        s.append(new_note)

        # Harmony stream.
        s_harmony = stream.Part()
        s_harmony.id = 'harmony'

        prev_harmony = set()
        new_chord = None
        num_frames = 0
        for idx, frame in enumerate(self.song):
            harmony = frame[HARMONY_INDICES_RANGE[0]:HARMONY_INDICES_RANGE[1]]
            pc_set = set()
            for pc, value in enumerate(harmony):
                if value:
                    # pc is a pitch class form 0 to 11. Convert to a pitch.
                    p = pitch.Pitch(pc)
                    p.octave = 3
                    pc_set.add(p)
            new_harmony = pc_set
            if new_harmony == prev_harmony:
                # same harmony continued
                num_frames += 1
                continue

            # If the set of pitch classes changes, commit previous chord and start new one.
            prev_harmony = new_harmony
            # commit the previous chord.
            if new_chord:
                new_chord.quarterLength = .25 * num_frames
                s_harmony.append(new_chord)
            # Start a new chord.
            if len(new_harmony) > 0:
                new_chord = chord.Chord(new_harmony)
            else:
                # rest
                new_chord = note.Rest()
            num_frames = 1

        # Commit the final note
        new_chord.quarterLength = .25 * num_frames
        s_harmony.append(new_chord)

        # Put melody and harmony lines together.
        score = stream.Score()
        score.insert(0, s)
        score.insert(0, s_harmony)
        #score.show()

        # Write midi file to disk.
        print 'Writing MIDI file: %s' % filename
        mf = midi.translate.streamToMidiFile(score.flat)
        mf.open(filename, 'wb')
        mf.write()
        mf.close()

if __name__ == '__main__':
    # Make a random song.
    random_song = np.random.randint(0, 2, size=(16, FEATURE_VECTOR_LENGTH))

    exporter = MidiExporter(random_song)
    exporter.create_midi_file('random.midi')


