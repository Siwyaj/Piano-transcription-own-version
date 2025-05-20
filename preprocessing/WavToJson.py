import os
import h5py
import sys
import csv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
import json

def CreateHdf5File():
    csv_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__),"..", 'dataset', 'maestro-v3.0.0', 'maestro-v3.0.0.csv'))
    with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        #print(f"csv_reader: {csv_reader}")
        csv_lines = list(csv_reader)
        print(f"csv_lines: {csv_lines[0]}")
        for line in tqdm.tqdm(csv_lines[1:]):
            #print(f"line: {line}")
            if line[2] == "train":
                wav_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'maestro-v3.0.0', line[5]))
                midi_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'maestro-v3.0.0', line[4]))
                print(f"Processing: {wav_file} with {midi_file}")
                #consider making the hdf5 group here and pars as an argument
                WavToJson(wav_file, midi_file, "hdf5Files/train_hdf5_file.h5")
            elif line[2] == "test":
                wav_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'maestro-v3.0.0', line[5]))
                midi_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'maestro-v3.0.0', line[4]))
                print(f"Processing: {wav_file} with {midi_file}")
                WavToJson(wav_file, midi_file, "hdf5Files/test_hdf5_file.h5")
            elif line[2] == "validation":
                wav_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'maestro-v3.0.0', line[5]))
                midi_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'maestro-v3.0.0', line[4]))
                print(f"Processing: {wav_file} with {midi_file}")
                WavToJson(wav_file, midi_file, "hdf5Files/val_hdf5_file.h5")
    pass

def WavToCqt(wav):
    import librosa
    import numpy as np
    # Load the audio file
    y, sr = librosa.load(wav, sr=config.sample_rate)
    # Compute the CQT
    cqt = librosa.cqt(y, sr=config.sample_rate, hop_length=config.hop_length, n_bins=config.n_bins, bins_per_octave=config.bins_per_octave)
    return cqt

def WavToJson(wav_file, h5_path):

    json_name = wav_file + ".json"
    cqt = WavToCqt(wav_file)
    # Save the CQT to a JSON file
    with open(json_name, 'w') as json_file:
        json_file.write(json.dumps(cqt.tolist()))
        print(f"Saved CQT to {json_name}")