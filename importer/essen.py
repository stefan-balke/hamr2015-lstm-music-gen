"""Importer for the Essen folk song collection
   http://www.esac-data.org/

"""
import numpy as np
from base import ImporterBase
import os
import re
from copy import copy
class Essen(ImporterBase):
    """Base Class for the dataset import.
    """

    def __init__(self, path='/home/anna/HAMR2015 - LSTM music/essen/all_songs'):
        self.output = []
        all_songs = os.listdir(path)
        for song in all_songs:
            song = all_songs[1]
            song = self.import_piano_roll(os.path.join(path, song))
            matrix = self.add_beat_flags(song[0], song[1], song[2], song[3])
            matrix = np.transpose(matrix)
            self.output.append(matrix)
        print len(self.output)
    def import_piano_roll(self, path_to_file):
        #diatonic scale to chromatic        
        #minus hundred is a rest 
        ds_to_ch = {0:-100, 1:0, 2:2, 3:4, 4:5, 5:7, 6:9, 7:11}
        
        off_beat_size = -1
        song_matrix = np.array([])
        lines = open(path_to_file).readlines()
        [melody, basic_unit, meter_units_per_measure, meter_unit] = self.parseFile(lines)
        if(meter_unit<0):
            return [song_matrix, -1, -1, -1] #we are not going to use songs with free meter for now
        multiplication_parameter = 16/basic_unit
        measures = melody.split(' ')
        for measure in measures:
            #32 for 3 octaves, 33 for continuity bit, 34 - 38 for position in measure 
            single_notes = self.split_into_single_notes(measure)
            for note in single_notes:
                bitmask = [0]*54
                #we parse a single note which is in format -7b__. 
                matchObj = re.match(r'([-+]?)(\d)([b#]?)(_*)(\.?)', note)
                if matchObj:
                    # octave modifier:   matchObj.group(1)
                    # pitch:  matchObj.group(2)
                    # sharp or flat: ' matchObj.group(3)
                    # duration:  matchObj.group(4)
                    # dot: matchObj.group(5) 
                    duration = 1
                    if(len(matchObj.group(4))>0):
                      duration = 2**len(matchObj.group(4))
                    if( matchObj.group(5) != ''):
                       duration *=1.5
                    pitchheight = 12+ds_to_ch[int(matchObj.group(2))]

                    if(matchObj.group(3) == 'b'):
                        pitchheight-=1
                    elif(matchObj.group(3) == '#'):
                        pitchheight+=1
                    
                    if(matchObj.group(1) == '+'):
                        pitchheight+=12
                    elif(matchObj.group(1) == '-'):
                        pitchheight-=12
                    
                    #now set the bitmask
                    if(pitchheight>0): #that is, if the note is not a rest
                        bitmask[pitchheight]=1
                    for i in range(int(duration)):
                        if(len(song_matrix)==0):
                            bitmask[48]=1 #set continuation bit that it's a new note                            
                            song_matrix = [copy(bitmask)]
                                                            
                        elif(len(song_matrix)>0 and i==0):   
                            bitmask[48]=1
                        else:
                            bitmask[48]=0
                        song_matrix.append(copy(bitmask))
            if(off_beat_size==-1): #we just parsed the first measure and don't know whether it was off-beat or not
                if(len(song_matrix)< meter_unit*(16/meter_units_per_measure)):
                    off_beat_size = len(song_matrix)
                else:
                    off_beat_size = 0
        return [song_matrix, off_beat_size, meter_unit*(16/meter_units_per_measure), 16/meter_units_per_measure]
        
        
    #still problematic: 
    #triplets (1_2_3_)
    # I don't know what ^ means
    def split_into_single_notes(self, measure):
        pattern = re.compile(r'[-+]?\d[b#]?_*\.?')
        m = re.findall(pattern, measure)
        return m
        #this is to check for more problematic cases
        #if(len(''.join(m))!=len(measure)):
        #    print measure
        #    print m
        
    def parseFile(self, lines):
        key_signature = ''
        melody = ''
        melody_on = False
        for l in lines:
            matchObj = re.match( r'^([A-Z]{3})\[(.*)\]?', l)
            if matchObj:
               if(matchObj.group(1) =='KEY'):
                   key_signature = matchObj.group(2)[:-1]
               elif(matchObj.group(1) =='MEL'):
                   melody = matchObj.group(2).strip()
                   melody_on = True
               elif(matchObj.group(1) == 'FCT'):
                   melody_on = False
            else:
               if(melody_on):
                   melody += l.strip()
        
        #remove the //] >> thing in the end of each melody
        matchObj = re.match( r'(.*)\s*//\] >>', melody)
        melody = matchObj.group(1)
        
        matchObj = re.match( r'.*\s*(\d\d)\s*[A-Za-z#b]*\s*((\d/\d)|FREI).*', key_signature)
        basic_unit = int(matchObj.group(1))
        meter = matchObj.group(2)
        if(meter != 'FREI'):
            matchObj = re.match( r'(\d+)/(\d+)', meter)
            meter_units_per_measure = int(matchObj.group(1))
            meter_unit = int(matchObj.group(2))
        else:
            meter_units_per_measure = -1
            meter_unit = -1

        return [melody, basic_unit, meter_units_per_measure, meter_unit]
    
    def add_beat_flags(self,  song_matrix, off_beat_size, measure_size, measure_unit):
        for ind, row in enumerate(song_matrix):
            if(row[48]==1):
                if(ind>=off_beat_size): #we are in the measure 1 or further
                    position = ((ind+1) - off_beat_size)%measure_size #find position in measure
                    number = self.get_metric_level_from_num_divisions(position, measure_size)
                    song_matrix[ind][49+number] = 1
        return song_matrix
    
