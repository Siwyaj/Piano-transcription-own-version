import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from preprocessing.WavSpecAndMidiSegmentor import wav_to_spec, midi_to_segmented_frames
from preprocessing.MidiLabelGenerator import segment_events_to_labels

class MaestroDataset(torch.utils.data.Dataset):
    def __init__(self, wav_paths, midi_paths, transform=None):
        self.wav_paths = wav_paths
        self.midi_paths = midi_paths
        self.segment_duration = config.secment_length
        self.transform = transform
        self.sample_rate = config.sample_rate
        self.hop_length = self.sample_rate // config.frames_per_second

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, idx):
        wav_file = self.wav_paths[idx]
        midi_file = self.midi_paths[idx]

        spectrograms, _ = wav_to_spec(wav_file)
        midi_segments = midi_to_segmented_frames(midi_file, self.hop_length)

        items = []
        for spec, midi_segment in zip(spectrograms, midi_segments):
            labels = segment_events_to_labels(midi_segment, spec.shape[1])

            if self.transform:
                spec = self.transform(spec)

            items.append((spec, labels))
        
        return items  # Optional: flatten this if needed
