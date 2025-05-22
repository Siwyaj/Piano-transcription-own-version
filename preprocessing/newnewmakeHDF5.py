import os
import csv
import h5py
import numpy as np
import librosa
import pretty_midi
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def encode_midi_to_label(midi_file, sr, segment_start, segment_end):
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    outputs = []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            if segment_start <= note.start < segment_end:
                outputs.append([note.start, note.end, note.pitch])
    outputs.sort(key=lambda x: x[0])
    return np.array(outputs, dtype=np.float32)

def process_and_store(wav_file, midi_file, h5_group, segment_duration=config.segment_duration, sr=config.sample_rate):
    audio, _ = librosa.load(wav_file, sr=sr)
    segment_idx = 0

    full_cqt = librosa.cqt(audio, sr=sr, hop_length=config.hop_len, n_bins=config.n_bins, bins_per_octave=config.bins_per_octave)
    full_cqt_mag = np.abs(full_cqt)
    full_cqt_db = librosa.amplitude_to_db(full_cqt_mag, ref=np.max)

    # Parameters
    segment_len = config.max_len  # 101 frames
    pad_len = 4                   # frames of padding before and after

    total_frames = full_cqt_db.shape[1]
    step = segment_len  # or smaller if overlapping segments

    for start_frame in range(pad_len, total_frames - pad_len - segment_len + 1, step):
        # slice segment with padding
        segment = full_cqt_db[:, start_frame - pad_len : start_frame + segment_len + pad_len]

        #segment = segment.T

        # label start/end time in seconds
        start_time = (start_frame * config.hop_len) / sr
        end_time = ((start_frame + segment_len) * config.hop_len) / sr
        frame_padding_time = config.hop_len * 4 / sr
        #print(f"frame padding time: {frame_padding_time}")
        label = encode_midi_to_label(midi_file, sr, start_time + frame_padding_time, end_time - frame_padding_time)
        #print(f"label shape:{label}")
        grp = h5_group.create_group(f"segment_{segment_idx}")
        grp.create_dataset("cqt", data=segment.astype(np.float32), compression="gzip")
        grp.create_dataset("label", data=label, dtype=np.float32)
        grp.attrs["start"] = start_time
        grp.attrs["end"] = end_time
        segment_idx += 1 


    """
    total_duration = librosa.get_duration(y=audio, sr=sr)
    This privously split the audio into segments and compute CQT for each segment
    instead for new paper, we will compute CQT for the whole audio and then split it into segments
    for start in np.arange(0, total_duration - segment_duration, segment_duration):
        end = start + segment_duration
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        audio_segment = audio[start_sample:end_sample]
        cqt_segment = librosa.cqt(audio_segment, sr=sr, hop_length=config.hop_len, n_bins=config.n_bins, bins_per_octave=config.bins_per_octave)
        #print(f"cqt shape: {cqt_segment.shape}")
        cqt_mag = np.abs(cqt_segment)
        cqt_db = librosa.amplitude_to_db(cqt_mag, ref=np.max)

        #print(f"[{os.path.basename(wav_file)}] Segment {segment_idx}: CQT shape = {cqt_db.shape}")


        label = encode_midi_to_label(midi_file, sr, start, end)

        grp = h5_group.create_group(f"segment_{segment_idx}")
        #grp.create_dataset("audio", data=audio_segment, dtype=np.float32)
        grp.create_dataset("cqt", data=cqt_db.astype(np.float32), compression="gzip")  # Store CQT here
        grp.create_dataset("label", data=label, dtype=np.float32)
        grp.attrs["start"] = start
        grp.attrs["end"] = end
        segment_idx += 1
        """

def create_hdf5():
    csv_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", 'dataset', 'maestro-v3.0.0', 'maestro-v3.0.0.csv'))

    # Prepare HDF5 files for each split
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'HDF5'))
    os.makedirs(output_dir, exist_ok=True)

    h5_files = {
        'train': h5py.File(os.path.join(output_dir, 'onetrain.hdf5'), 'w'),
        'test': h5py.File(os.path.join(output_dir, 'onetest.hdf5'), 'w'),
        'validation': h5py.File(os.path.join(output_dir, 'onevalidation.hdf5'), 'w')
    }

    with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        csv_lines = list(csv_reader)

        for i, line in enumerate(tqdm(csv_lines[1:2])):  # Limit for testing
            split = line[2]
            if split in h5_files:
                wav_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'maestro-v3.0.0', line[5]))
                midi_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'maestro-v3.0.0', line[4]))
                print(f"Processing {split}: {wav_file}")
                file_group = h5_files[split].create_group(f"file_{i}")
                process_and_store(wav_file, midi_file, file_group)

    for f in h5_files.values():
        f.close()

if __name__ == "__main__":
    create_hdf5()
