"""
Importer for the Essen folk song collection that does not transpose the songs

"""
from essen import ImporterEssen
import re
import numpy as np
from copy import copy
import random 

class EssenUntransposed(ImporterEssen):
    """Base Class for the dataset import.
    """
    
    def import_piano_roll(self, path_to_file):
        # diatonic scale to chromatic
        # minus hundred is a rest
        ds_to_ch = {0:-100, 1:0, 2:2, 3:4, 4:5, 5:7, 6:9, 7:11}
        random_transposition = int(random.uniform(-6,6))
        
        off_beat_size = -1
        song_matrix = np.array([])
        lines = open(path_to_file).readlines()
        [melody, basic_unit, meter_units_per_measure, meter_unit] = self.parseFile(lines)
        if(meter_unit<0):
            return [song_matrix, -1, -1, -1]  # we are not going to use songs with free meter for now
        multiplication_parameter = 16/basic_unit
        measures = melody.split('  ')
        for measure in measures:
            measure_matrix = []
            # 32 for 3 octaves, 33 for continuity bit, 34 - 38 for position in measure
            single_notes = self.split_into_single_notes(measure)
            for note in single_notes:
                bitmask = [0]*54
                # we parse a single note which is in format -7b__.
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
                    
                    pitchheight = 12+random_transposition +ds_to_ch[int(matchObj.group(2))]
                    #pitchheight = 12 +ds_to_ch[int(matchObj.group(2))]
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
                            measure_matrix = [copy(bitmask)]
                            continue      
                        elif(len(song_matrix)>0 and i==0):   
                            bitmask[48]=1
                        else:
                            bitmask[48]=0
                        measure_matrix.append(copy(bitmask))
            if(len(measure_matrix) ==0): #if the measure is empty, probably there are just pauses in it and we throw it away
                continue
            measure_matrix = self.add_harmony(measure_matrix)
            if(len(song_matrix)==0):
                song_matrix = measure_matrix
            else:
                song_matrix = song_matrix + measure_matrix
            if(off_beat_size==-1): #we just parsed the first measure and don't know whether it was off-beat or not
                if(len(song_matrix)< meter_unit*(16/meter_units_per_measure)):
                    off_beat_size = len(song_matrix)
                else:
                    off_beat_size = 0
            
        return [song_matrix, off_beat_size, meter_units_per_measure*(16/meter_unit), 16/meter_units_per_measure]