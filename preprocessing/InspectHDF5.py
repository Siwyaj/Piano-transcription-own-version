import h5py
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import sys
import os

# Assuming config.py is in the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def plot_cqt_with_onsets_all(hdf5_path, sr=config.sample_rate, hop_length=config.hop_len):
    with h5py.File(hdf5_path, "r") as f:
        for file_group_name in f.keys():
            file_group = f[file_group_name]
            
            for segment_name in file_group.keys():
                segment = file_group[segment_name]
                
                cqt_db = segment["cqt"][:]  # CQT spectrogram in dB
                print(f"cqt_db shape: {cqt_db.shape}")
                label = segment["onset_label"][:]  # Binary matrix: (n_frames, 128)
                segment_start = segment.attrs["start"]

                if np.sum(label) == 0:
                    print(f"Skipping {file_group_name}/{segment_name} — no onsets.")
                    continue

                plt.figure(figsize=(14, 6))

                # Plot CQT spectrogram
                librosa.display.specshow(
                    cqt_db, sr=sr, x_axis='time', y_axis='cqt_note', hop_length=hop_length
                )
                plt.colorbar(format="%+2.0f dB")
                plt.title(f"{file_group_name}/{segment_name} — Segment start: {segment_start:.2f}s")

                # Detect onsets: rising edges in the binary label matrix
                onset_frames = np.logical_and(
                    label > 0,
                    np.pad(label[:-1, :] == 0, ((1, 0), (0, 0)), mode='constant')
                )
                onset_indices = np.argwhere(onset_frames)
                onset_times = onset_indices[:, 0] * hop_length / sr  # frame to time

                print(f"Detected {len(onset_times)} note onsets")

                # Plot vertical lines for onsets
                for t in onset_times:
                    if 0 <= t <= cqt_db.shape[1] * hop_length / sr:
                        plt.axvline(t, color='lime', linestyle='--', linewidth=1.5, alpha=0.7)

                plt.tight_layout()
                plt.show()

def print_hdf5_structure(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        def print_group(name, obj):
            if isinstance(obj, h5py.Group):
                print(f"Group: {name}")
                for key in obj.keys():
                    print_group(f"{name}/{key}", obj[key])
            elif isinstance(obj, h5py.Dataset):
                print(f"Dataset: {name}, shape: {obj.shape}, dtype: {obj.dtype}")

        f.visititems(print_group)

# Example usage
hdf5_path = "HDF5/t_101_n_5train.hdf5"
plot_cqt_with_onsets_all(hdf5_path)

# Optional: check HDF5 structure
# hdf5_path = "HDF5/t_101_n_10train.hdf5"
# print_hdf5_structure(hdf5_path)
