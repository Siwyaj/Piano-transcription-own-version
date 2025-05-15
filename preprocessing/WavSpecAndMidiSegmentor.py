'''
This script is used to convert audio and MIDI files into a structured HDF5 format with mel spectrograms and MIDI targets. It processes audio files, extracts mel spectrograms, and generates MIDI targets for each segment of the audio. The resulting data is stored in an HDF5 file for efficient access during training or evaluation of machine learning models.
'''
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import librosa
import pretty_midi
import numpy as np
import os
import config
import h5py
import tqdm

def midi_to_targets(midi, num_frames, segment_start_time, segment_duration):
    onset_targets = np.zeros((num_frames, 88), dtype=np.float32)
    frame_targets = np.zeros((num_frames, 88), dtype=np.float32)
    velocity_targets = np.zeros((num_frames, 88), dtype=np.float32)

    segment_end_time = segment_start_time + segment_duration

    for note in midi.instruments[0].notes:
        if note.end < segment_start_time or note.start > segment_end_time:
            continue

        note_start = max(note.start, segment_start_time) - segment_start_time
        note_end = min(note.end, segment_end_time) - segment_start_time

        start_frame = int(note_start * config.frames_per_second)
        end_frame = int(note_end * config.frames_per_second)

        pitch_index = note.pitch - 21
        if not (0 <= pitch_index < 88):
            continue

        if 0 <= start_frame < num_frames:
            onset_targets[start_frame, pitch_index] = 1.0
            velocity_targets[start_frame, pitch_index] = note.velocity / 127.0

        frame_targets[start_frame:end_frame, pitch_index] = 1.0

    return onset_targets, frame_targets, velocity_targets


def wav_to_spec(wav_file, midi_file, hdf5_path, segment_duration=config.secment_length, sr=config.sample_rate, segment_index_start=0):
    tqdm.tqdm.write(f"Processing: {wav_file} with {midi_file}")
    y, _ = librosa.load(wav_file, sr=sr)
    hop_length = sr // config.frames_per_second
    segment_samples = segment_duration * sr
    num_segments = len(y) // segment_samples

    midi = pretty_midi.PrettyMIDI(midi_file)
    current_segment = segment_index_start

    with h5py.File(hdf5_path, 'a') as hdf5_file:  # 'a' = append mode
        for i in range(num_segments):
            start_sample = i * segment_samples
            end_sample = (i + 1) * segment_samples
            segment_start_time = i * segment_duration

            audio_segment = y[start_sample:end_sample]
            mel_spectrogram = librosa.feature.melspectrogram(
                y=audio_segment, sr=sr,
                n_fft=config.window_size,
                hop_length=hop_length,
                n_mels=config.mel_bins
            )
            mel_spectrogram = librosa.power_to_db(mel_spectrogram)
            num_frames = mel_spectrogram.shape[1]

            # Ensure mel_spectrogram shape is (256, 1001)
            expected_shape = (config.mel_bins, 1001)
            mel_bins, time_bins = mel_spectrogram.shape

            if mel_bins != config.mel_bins:
                tqdm.tqdm.write(f"Warning: Unexpected mel bin count: {mel_bins}, expected {config.mel_bins}")
                continue  # skip corrupt data

            if time_bins < expected_shape[1]:
                # Pad right with lowest dB value
                pad_width = expected_shape[1] - time_bins
                mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, pad_width)), mode='constant', constant_values=-80.0)
            elif time_bins > expected_shape[1]:
                # Truncate
                mel_spectrogram = mel_spectrogram[:, :expected_shape[1]]


            onset, frame, velocity = midi_to_targets(
                midi, num_frames, segment_start_time, segment_duration)

            group_name = f"{wav_file}_segment_{current_segment}"
            if group_name in hdf5_file:
                print(f"Warning: Group {group_name} already exists. Overwriting.")
                del hdf5_file[group_name]  #delete existing group if it exists

            segment_group = hdf5_file.create_group(group_name)
            segment_group.create_dataset('mel', data=mel_spectrogram, compression="gzip")
            segment_group.create_dataset('onset', data=onset, compression="gzip")
            segment_group.create_dataset('frame', data=frame, compression="gzip")
            segment_group.create_dataset('velocity', data=velocity, compression="gzip")

            current_segment += 1

    return current_segment  # So you can track the next index


if __name__ == "__main__":
    audio_file = 'test.wav'
    midi_file = 'test.midi'
    hdf5_path = 'test.h5'

    segment = wav_to_spec(audio_file, midi_file, hdf5_path)

    print(f"Segmented {segment} segments.")
