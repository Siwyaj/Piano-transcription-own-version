import WavSpecAndMidiSegmentor
import os
import h5py
import sys
import csv
import tqdm
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
                WavSpecAndMidiSegmentor.wav_to_spec(wav_file, midi_file, "hdf5Files/train_hdf5_file")
            elif line[2] == "test":
                wav_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'maestro-v3.0.0', line[5]))
                midi_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'maestro-v3.0.0', line[4]))
                print(f"Processing: {wav_file} with {midi_file}")
                WavSpecAndMidiSegmentor.wav_to_spec(wav_file, midi_file, "hdf5Files/test_hdf5_file")
            elif line[2] == "validation":
                wav_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'maestro-v3.0.0', line[5]))
                midi_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'maestro-v3.0.0', line[4]))
                print(f"Processing: {wav_file} with {midi_file}")
                WavSpecAndMidiSegmentor.wav_to_spec(wav_file, midi_file, "hdf5Files/val_hdf5_file.h5")
    pass


def ReadAndPrintHdf5File(hdf5_path="hdf5Files/train_hdf5_file"):
    with h5py.File(hdf5_path, 'r') as hdf5_file:
        for key in hdf5_file.keys():
            print(f"Key: {key}")
            for sub_key in hdf5_file[key].keys():
                print(f"  Sub-key: {sub_key}")

    pass

if __name__ == "__main__":
    #CreateHdf5File()
    ReadAndPrintHdf5File()