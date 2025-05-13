'''
This script will take a wav file input and convert it to a spectrogram using librosa
'''

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import librosa
import config
import pretty_midi
import numpy as np


def wav_to_spec(wav_file, segment_duration=config.secment_length, sr=config.sample_rate):
    # Load the wav file
    print(f"Loading wav file: {wav_file}")
    y, _ = librosa.load(wav_file, sr=sr)

    # Define the hop_length (based on the sample rate and frames per second)
    hop_length = sr // config.frames_per_second
    #print(f"hop_length: {hop_length}")

    # Segment the audio into chunks of length segment_duration (in seconds)
    segment_samples = segment_duration * sr  # Number of samples per segment
    num_segments = len(y) // segment_samples  # Total number of full segments
    #print(f"y.shape: {len(y)}")
    #print(f"num_segments: {num_segments}")
    spectrograms = []  # List to store individual spectrograms

    for i in range(num_segments):
        start_sample = i * segment_samples
        end_sample = (i + 1) * segment_samples
        #print(f"start_sample: {start_sample}, end_sample: {end_sample}")
        audio_segment = y[start_sample:end_sample]

        # Convert the audio segment to a Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=audio_segment, sr=sr, n_fft=config.window_size, hop_length=hop_length, n_mels=config.mel_bins)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram)
        
        # Append the Mel spectrogram to the list
        spectrograms.append(mel_spectrogram)

    return spectrograms, hop_length

import pretty_midi

def midi_to_segmented_frames(midi_file, hop_length, segment_duration=config.secment_length, sr=config.sample_rate):
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    frame_time = hop_length / sr
    frames_per_segment = int(segment_duration / frame_time)
    total_frames = int(midi_data.get_end_time() / frame_time)
    num_segments = (total_frames // frames_per_segment) + 1

    segments = [[] for _ in range(num_segments)]

    for instrument in midi_data.instruments:
        if instrument.is_drum:
            continue

        for note in instrument.notes:
            start_frame = int(note.start / frame_time)
            end_frame = int(note.end / frame_time)
            pitch = note.pitch
            velocity = note.velocity

            # Determine which segment this note starts in
            segment_index = start_frame // frames_per_segment
            if segment_index >= len(segments):
                continue

            # Local frame indices within the segment
            local_start = start_frame % frames_per_segment
            local_end = min(end_frame, (segment_index + 1) * frames_per_segment) % frames_per_segment

            # Ignore if clipped duration becomes 0
            if local_start >= local_end:
                continue

            segments[segment_index].append((local_start, local_end, pitch, velocity))

    return segments




if __name__ == "__main__":
    #test the function
    import matplotlib.pyplot as plt

    audio_file = 'test.wav'
    midi_file = 'test.midi'

    mel_spectrogram, hop_length = wav_to_spec(audio_file)
    print(mel_spectrogram[0])
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(mel_spectrogram[0], x_axis='time', y_axis='mel', sr=config.sample_rate)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    #plt.show() #outcomment this line to show the plots, waring: this will show a lot of plots    
    midievent = midi_to_segmented_frames(midi_file, hop_length)
    print(midievent[0])
    print(mel_spectrogram[0].shape[1])