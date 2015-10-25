import os
import sys
import numpy as np
from sqlalchemy import Column, ForeignKey, Integer, Float, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import csv
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


def get_solo_pitch_shape(solo, frame_times, n_pitch_classes=None, transposition_offset=0):
    if n_pitch_classes:
        n_pitches = n_pitch_classes
    else:
        n_pitches = 120

    solo_length = solo.melodies[-1].onset + solo.melodies[-1].duration
    solo_piano_roll = np.zeros((n_pitches, len(frame_times)))

    for note_event in solo.melodies:
        idx_start = np.argmin(np.abs(frame_times-note_event.onset))
        idx_end = np.argmin(np.abs(frame_times-(note_event.onset+note_event.duration)))

        if n_pitch_classes:
            cur_pitch = (note_event.pitch-transposition_offset) % n_pitch_classes
        else:
            cur_pitch = note_event.pitch-transposition_offset

        cur_pitch_vector = np.zeros((n_pitches, 1))
        cur_pitch_vector[cur_pitch] = 1.0
        solo_piano_roll[:, idx_start:idx_end] = cur_pitch_vector

    return solo_piano_roll


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
    def __init__(self):
        self.output = []
        self.pr_n_pitches = 36
        self.pr_bar_division = 16

        melids = get_melids()

        for cur_melid in melids:
            self.output.append(self.import_piano_roll(cur_melid))

    def import_piano_roll(self, cur_melid):
        solo = get_solo(cur_melid)
        transp_offset = get_transposition_offset(solo)

        beats = get_solo_beats(solo, 2)
        solo_piano_roll = get_solo_pitch_shape(solo, beats, n_pitch_classes=36, transposition_offset=transp_offset)

        return solo_piano_roll

    def add_beat_flags(self):
        pass

if __name__ == '__main__':
    importer = ImporterWJD()

    print 'test'