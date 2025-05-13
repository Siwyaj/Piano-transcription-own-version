import os
import sys
import csv

def GetWavAndMidiPathFromcsv(maestro_CSV_path, split='train'):
    wav_paths = []
    midi_paths = []
    with open(maestro_CSV_path, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        #print(f"csv_reader: {csv_reader}")
        csv_lines = list(csv_reader)
        print(f"csv_lines: {csv_lines[0]}")
        for line in csv_lines[1:]:
            #print(f"line: {line}")
            if line[2] == split:

                wav_paths.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'maestro-v3.0.0',line[5])))
                midi_paths.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'maestro-v3.0.0',line[4])))
    return wav_paths, midi_paths

if __name__ == "__main__":
    maestro_CSV_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'..', 'dataset','maestro-v3.0.0', 'maestro-v3.0.0.csv'))
    wav_paths, midi_paths = GetWavAndMidiPathFromcsv(maestro_CSV_path)
    #print(f"wav_paths: {wav_paths[0]}")
    #print(f"midi_paths: {midi_paths[0]}")