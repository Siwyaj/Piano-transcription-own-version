import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config


def midi_to_pianoroll(label, segment_duration, frame_hop, n_pitches=88):
    """
    Convert MIDI note info (start, end, pitch) into frame-wise piano roll:
    shape (time_frames, n_pitches), binary 0/1 for note active or not.
    """
    time_frames = int(segment_duration / frame_hop)
    pianoroll = np.zeros((time_frames, n_pitches), dtype=np.float32)
    
    for start, end, pitch in label:
        start_frame = int(start / frame_hop)
        end_frame = int(end / frame_hop)
        if pitch < 21 or pitch >= 21 + n_pitches:
            continue  # ignore notes outside piano range A0 (21) to C8 (108)
        pitch_idx = int(pitch) - 21
        start_frame = max(0, start_frame)
        end_frame = min(time_frames - 1, end_frame)
        pianoroll[start_frame:end_frame+1, pitch_idx] = 1.0
    return pianoroll

class PianoDataset(Dataset):
    def __init__(self, hdf5_path):
        self.h5 = h5py.File(hdf5_path, 'r')
        self.samples = []
        # Collect all segment paths (e.g., file_0/segment_0)
        for file_key in self.h5.keys():
            for seg_key in self.h5[file_key].keys():
                self.samples.append(f"{file_key}/{seg_key}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        segment_path = self.samples[idx]
        grp = self.h5[segment_path]
        cqt = grp['cqt'][()]          # shape (freq_bins, time_frames)
        label = grp['label'][()]      # shape (N, 3) start, end, pitch
        
        # Convert MIDI notes to pianoroll frame labels
        pianoroll = midi_to_pianoroll(label, segment_duration=config.segment_duration, frame_hop=config.frame_hop, n_pitches=config.n_pitches)
        
        # Normalize CQT (log scale)
        cqt = np.log1p(np.abs(cqt))
        
        # Convert to tensor, add channel dim (for CNN)
        cqt_tensor = torch.tensor(cqt, dtype=torch.float32).unsqueeze(0)  # (1, freq_bins, time_frames)
        label_tensor = torch.tensor(pianoroll, dtype=torch.float32)       # (time_frames, n_pitches)
        
        return cqt_tensor, label_tensor
