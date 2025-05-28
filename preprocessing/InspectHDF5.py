import h5py
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import sys
import os
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
                label = segment["onset_label"][:]  # MIDI labels: [start, end, pitch]
                segment_start = segment.attrs["start"]

                if label.shape[0] == 0:
                    print(f"Skipping {file_group_name}/{segment_name} — no onsets.")
                    continue

                plt.figure(figsize=(14, 6))

                # Plot CQT spectrogram
                librosa.display.specshow(cqt_db, sr=sr, x_axis='time', y_axis='cqt_note', hop_length=hop_length)
                plt.colorbar(format="%+2.0f dB")
                plt.title(f"{file_group_name}/{segment_name} — Segment start: {segment_start:.2f}s")

                # Plot onsets
                note_onsets = label[:, 0] - segment_start
                for onset in note_onsets:
                    if 0 <= onset <= cqt_db.shape[1] * hop_length / sr:
                        plt.axvline(onset, color='g', linestyle='--', alpha=0.7)

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
hdf5_path = "HDF5/onetrain.hdf5"
#plot_cqt_with_onsets_all(hdf5_path)
hdf5_path = "HDF5/t_101_n_10train.hdf5"
print_hdf5_structure(hdf5_path)

