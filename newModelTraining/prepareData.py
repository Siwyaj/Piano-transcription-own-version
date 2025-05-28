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

import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import config

import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import config

class PianoDataset(Dataset):
    def __init__(self, hdf5_path, label_type='both'):
        """
        label_type: 'onset', 'sustain', or 'both'
        """
        self.h5 = h5py.File(hdf5_path, 'r')
        self.samples = []
        self.label_type = label_type

        for file_key in self.h5.keys():
            for seg_key in self.h5[file_key].keys():
                self.samples.append(f"{file_key}/{seg_key}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        segment_path = self.samples[idx]
        grp = self.h5[segment_path]
        
        # Load CQT and normalize
        cqt = grp['cqt'][()]  # shape: (freq_bins, time_frames)
        cqt = np.log1p(np.abs(cqt))
        cqt_tensor = torch.tensor(cqt, dtype=torch.float32).unsqueeze(0)  # (1, freq_bins, time_frames)

        # Load label(s)
        if self.label_type == 'onset':
            label = grp['onset_label'][()]  # (time_frames, 88)
        elif self.label_type == 'sustain':
            label = grp['sustain_label'][()]  # (time_frames, 88)
        elif self.label_type == 'both':
            onset = grp['onset_label'][()]
            sustain = grp['sustain_label'][()]
            label = np.stack([onset, sustain], axis=0)  # (2, time_frames, 88)
        else:
            raise ValueError(f"Unsupported label_type: {self.label_type}")

        label_tensor = torch.tensor(label, dtype=torch.float32)

        onset_label = grp['onset_label'][()]
        sustain_label = grp['sustain_label'][()]

        return cqt_tensor, onset_label, sustain_label
