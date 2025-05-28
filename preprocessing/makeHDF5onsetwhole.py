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

def encode_midi_to_label(midi_file, sr, segment_start, segment_end, n_frames, hop_len, n_pitches=88):
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    
    # Initialize label matrices: (frames, pitches)
    onset_label = np.zeros((n_frames, n_pitches), dtype=np.float32)
    sustain_label = np.zeros((n_frames, n_pitches), dtype=np.float32)
    #print(f"onset_label shape: {onset_label.shape}")
    #print(f"sustain_label shape: {sustain_label.shape}")
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            # Only consider notes in this segment window (allow partial overlaps)
            if note.end < segment_start or note.start > segment_end:
                continue
            
            # Clamp note start/end to segment boundaries
            start = max(note.start, segment_start)
            end = min(note.end, segment_end)
            
            # Convert times to frame indices relative to segment start
            start_frame = int((start - segment_start) * sr / hop_len)
            end_frame = int((end - segment_start) * sr / hop_len)
            
            pitch_idx = note.pitch - 21  # MIDI note 21 = A0 (lowest piano key)
            if 0 <= pitch_idx < n_pitches:
                # Mark onset
                if 0 <= start_frame < n_frames:
                    onset_label[start_frame-5:start_frame+5, pitch_idx] = 1
                
                # Mark sustain from onset frame to offset frame
                if end_frame > n_frames:
                    end_frame = n_frames
                if end_frame > start_frame:
                    sustain_label[start_frame:end_frame, pitch_idx] = 1
    
    return onset_label, sustain_label


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
        n_frames = segment_len  # number of frames per segment (without padding)


        onset_label, sustain_label = encode_midi_to_label(
            midi_file,
            sr,
            start_time,
            end_time,
            n_frames=n_frames,
            hop_len=config.hop_len,
            n_pitches=88
        )
        grp = h5_group.create_group(f"segment_{segment_idx}")

        #print(f"label shape:{label}")
        grp.create_dataset("cqt", data=segment.astype(np.float32), compression="gzip")
        grp.create_dataset("onset_label", data=onset_label, compression="gzip")
        grp.create_dataset("sustain_label", data=sustain_label, compression="gzip")
        grp.attrs["start"] = start_time
        grp.attrs["end"] = end_time
        segment_idx += 1 

import os
import csv
import h5py
from tqdm import tqdm

def create_hdf5():
    csv_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", 'dataset', 'maestro-v3.0.0', 'maestro-v3.0.0.csv'))

    # Prepare HDF5 file for training split only
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'HDF5'))
    os.makedirs(output_dir, exist_ok=True)

    h5_file = h5py.File(os.path.join(output_dir, 'tolerance+-5_t_101_n_5train.hdf5'), 'w')

    count = 0
    max_files = 5

    with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        csv_lines = list(csv_reader)

        for i, line in enumerate(csv_lines[1:]):
            split = line[2]
            if split != 'train':
                continue
            wav_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'maestro-v3.0.0', line[5]))
            midi_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'maestro-v3.0.0', line[4]))
            print(f"Processing train file {count+1}: {wav_file}")
            file_group = h5_file.create_group(f"file_{count}")
            process_and_store(wav_file, midi_file, file_group)
            count += 1
            if count >= max_files:
                break

    h5_file.close()

if __name__ == "__main__":
    create_hdf5()
