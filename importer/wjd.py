import os
import sys
import numpy as np
from sqlalchemy import Column, ForeignKey, Integer, Float, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import csv
import settings
from base import ImporterBase

Base = declarative_base()


class Song(Base):
    __tablename__ = 'song'
    songid = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    filename_track = Column(String, nullable=False)
    solos = relationship('Solo')


class Solo(Base):
    __tablename__ = 'solo_info'
    melid = Column(Integer, primary_key=True)
    songid = Column(Integer, ForeignKey('song.songid'))
    melodies = relationship('Melody', backref='solo_info')
    beats = relationship('Beat', backref='solo_info')
    performer = Column(String)
    title = Column(String)
    instrument = Column(String)
    key = Column(String)
    signature = Column(String)


class Beat(Base):
    __tablename__ = 'beats'
    beatid = Column(Integer, primary_key=True)
    melid = Column(Integer, ForeignKey('solo_info.melid'))
    onset = Column(Float)


# class Section(Base):
#     __tablename__ = 'sections'
#     melid = Column(Integer, ForeignKey('solo_info.melid'))
#     type = Column(String)
#     start = Column(Integer)
#     end = Column(Integer)
#     value = Column(String)


class Melody(Base):
    __tablename__ = 'melody'
    eventid = Column(Integer, primary_key=True)
    melid = Column(Integer, ForeignKey('solo_info.melid'))
    onset = Column(Float)
    pitch = Column(Float)
    duration = Column(Float)


def get_melids():
    engine = create_engine('sqlite:///../data/wjazzd_new.db')
    db_session = sessionmaker(bind=engine)
    session = db_session()
    melids_query_gen = session.query(Solo.melid).distinct()

    melids = []

    for cur_melid in melids_query_gen:
        melids.append(cur_melid)

    return melids


def get_solo(melid):
    engine = create_engine('sqlite:///../data/wjazzd_new.db')
    db_session = sessionmaker(bind=engine)
    session = db_session()
    solo = session.query(Solo).get(melid)

    return solo


def get_solo_activity(melid, frame_times):
    solo = get_solo(melid)
    solo_length = solo.melodies[-1].onset + solo.melodies[-1].duration
    solo_activity = np.zeros_like(frame_times)

    for note_event in solo.melodies:
        idx_start = np.argmin(np.abs(frame_times-note_event.onset))
        idx_end = np.argmin(np.abs(frame_times-(note_event.onset+note_event.duration)))
        solo_activity[idx_start:idx_end] = 1.0

    return solo_activity


def get_solo_beats(solo, subdivisions=0):
    """

    Parameter
    ---------
    melid : integer
    subdivisions : integer, optional
        Defaults to 0.

    Return
    ------
    beats : ndarray

    """
    # include first and second note
    subdivisions_mod = subdivisions + 1
    beats = np.zeros(1)
    for cur_beat, note_event in enumerate(solo.beats):
        if subdivisions == 0:
            beats[cur_beat] = note_event.onset
        else:
            if cur_beat == 0:
                beats[0] = note_event.onset
            else:
                last_onset = beats[-1]
                cur_onset = note_event.onset

                # fill with subdivisions
                subdivsions_onsets = np.linspace(last_onset, cur_onset, subdivisions_mod)
                beats = np.r_[beats, subdivsions_onsets]

    return beats


def get_transposition_offset(solo):
    # define musical pitch classes
    pitch_classes_sharp = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'A', 'A#', 'B']
    pitch_classes_flat = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'A', 'Bb', 'B']

    # split the string
    cur_key = solo.key.split('-')[0]

    transp_offset = None

    # find it
    try:
        transp_offset = pitch_classes_sharp.index(cur_key)
    except ValueError:
        pass

    try:
        transp_offset = pitch_classes_flat.index(cur_key)
    except ValueError:
        pass

    # this means there was no annotation in the database
    if not transp_offset:
        transp_offset = 0

    return transp_offset


def visualize_piano_roll(piano_roll):
    import matplotlib.pyplot as plt
    plt.imshow(piano_roll, cmap=plt.get_cmap('gray_r'))


class ImporterWJD(ImporterBase):
    """Base Class for the dataset import.
    """
    def __init__(self, beats_per_measure, melody_range, harmony_range, continuation_range, metric_range, path='../data/rock_corpus_v2-1/rs200_melody_nlt'):
        self.output = []
        super(ImporterWJD, self).__init__(beats_per_measure, melody_range, harmony_range, continuation_range, metric_range)
        self.path = path
        self.output = []

        #'pr' stands for piano roll
        self.pr_n_pitches = melody_range[1] - melody_range[0]
        self.pr_width = self.metric_range[1]
        self.pr_bar_division = beats_per_measure

        melids = get_melids()

        for cur_melid in melids:
            self.output.append(self.import_piano_roll(cur_melid))

    def get_solo_pitch_shape(self, solo, frame_times, n_pitch_classes, transposition_offset):
        if n_pitch_classes:
            n_pitches = n_pitch_classes
        else:
            n_pitches = 120

        solo_length = solo.melodies[-1].onset + solo.melodies[-1].duration
        solo_piano_roll = np.zeros((self.pr_width, len(frame_times)))

        pitch_range_start = np.min([mel.pitch for mel in solo.melodies])
        pitch_range_end = np.max([mel.pitch for mel in solo.melodies])

        lowest_octave = int((pitch_range_start - transposition_offset) / 12) * 12
        for note_event in solo.melodies:
            note_metric_index = (note_event.beat - 1) * 4 + note_event.tatum - 1
            cur_bar_idx_start = note_event.bar * self.pr_bar_division
            idx_start = np.argmin(np.abs(frame_times-note_event.onset))
            idx_end = np.argmin(np.abs(frame_times-(note_event.onset+note_event.duration)))

            cur_metric_level = self.get_metric_level_from_num_divisions(metric_index, self.pr_bar_division)
            if n_pitch_classes:
                cur_pitch = (note_event.pitch-transposition_offset - lowest_octave) % n_pitch_classes
            else:
                cur_pitch = note_event.pitch-transposition_offset - lowest_octave

            cur_pitch_vector = np.zeros((self.pr_width, 1))
            cur_pitch_vector[cur_pitch] = 1.0
            solo_piano_roll[:, idx_start:idx_end] = cur_pitch_vector
            solo_piano_roll[self.metric_range[0] + cur_metric_level, cur_bar_idx_start+note_metric_index] = 1
        return solo_piano_roll


    def import_piano_roll(self, cur_melid):
        solo = get_solo(cur_melid)
        transp_offset = get_transposition_offset(solo)

        beats = get_solo_beats(solo, 2)
        solo_piano_roll = self.get_solo_pitch_shape(solo, beats, self.num_pitches, transp_offset)

        return solo_piano_roll

    def add_beat_flags(self):
        pass

if __name__ == '__main__':
    importer = ImporterWJD(settings.BEATS_PER_MEASURE, settings.MELODY_INDICES_RANGE, settings.HARMONY_INDICES_RANGE, settings.CONTINUATION_FLAG_RANGE, settings.METRIC_FLAGS_RANGE)

    print 'test'