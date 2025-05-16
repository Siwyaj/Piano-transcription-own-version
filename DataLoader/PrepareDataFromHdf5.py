'''
This script will prepare the data for the model.
It will load the hdf5 file and use pytorch's dataloader to load the data.
'''
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class PianoTranscriptionDataset(torch.utils.data.Dataset):
    def __init__(self, hdf5_path):
        self.hdf5_path = hdf5_path
        with h5py.File(hdf5_path, 'r') as f:
            self.keys = list(f.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        with h5py.File(self.hdf5_path, 'r') as f:
            group = f[key]
            mel = group['mel'][:]  # (229, T)
            onset = group['onset'][:]  # (T, 88)
            frame = group['frame'][:]  # (T, 88)
            velocity = group['velocity'][:]  # (T, 88)
            offset = group['offset'][:]  # (T, 88)


        mel = mel.T  # → (T, 229)
        onset = onset.astype(np.float32)
        frame = frame.astype(np.float32)
        velocity = velocity.astype(np.float32)

        # Convert to torch tensors
        mel = torch.tensor(mel, dtype=torch.float32)        # (T, 229)
        onset = torch.tensor(onset, dtype=torch.float32)    # (T, 88)
        frame = torch.tensor(frame, dtype=torch.float32)    # (T, 88)
        velocity = torch.tensor(velocity, dtype=torch.float32)  # (T, 88)
        offset = torch.tensor(offset, dtype=torch.float32)  # (T, 88)

        labels = {
            'onset': onset,
            'offset': offset,
            'velocity': velocity,
            'frame': frame
        }

        return mel, labels



def DataLoaderHdf5(hdf5_path="hdf5Files/train_hdf5_file", batch_size=16, shuffle=True, num_workers=8):
    dataset = PianoTranscriptionDataset(hdf5_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


if __name__ == "__main__":
    dataloader = DataLoaderHdf5("hdf5Files/train_hdf5_file")
    for mel, labels in dataloader:
        print("Mel shape:", mel.shape)
        print("Onset shape:", labels["onset"].shape)
        print("Frame shape:", labels["frame"].shape)
        break
