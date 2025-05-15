import torch
import torchaudio
import pretty_midi
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
from DataLoader.GetWavAndMidiPathFromcsv import GetWavAndMidiPathFromcsv
import os
import sys

import torch
import torchaudio
import pretty_midi

import torch
import torchaudio
import pretty_midi
import numpy as np

class MAESTRODataset(Dataset):
    def __init__(self, audio_paths, midi_paths, target_length=5000, transform=None):
        self.audio_paths = audio_paths
        self.midi_paths = midi_paths
        self.target_length = target_length  # Fixed length for target sequences
        self.transform = transform
    
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        # Load Audio (WAV file)
        waveform, sample_rate = torchaudio.load(self.audio_paths[idx])
        mel_spectrogram = torchaudio.transforms.MelSpectrogram()(waveform)

        # Pad or truncate the mel-spectrogram
        mel_spectrogram = self.pad_or_truncate(mel_spectrogram)

        # Load MIDI (pretty_midi)
        midi_file = pretty_midi.PrettyMIDI(self.midi_paths[idx])
        
        # Extract MIDI features like onset, offset, velocity, and pitch
        notes = midi_file.instruments[0].notes
        onsets = [note.start for note in notes]
        offsets = [note.end for note in notes]
        velocities = [note.velocity for note in notes]
        pitches = [note.pitch for note in notes]

        # Pad or truncate target sequences to the fixed length
        onsets = self.pad_or_truncate_target(onsets)
        offsets = self.pad_or_truncate_target(offsets)
        velocities = self.pad_or_truncate_target(velocities)
        pitches = self.pad_or_truncate_target(pitches)

        targets = {
            'onsets': torch.tensor(onsets),
            'offsets': torch.tensor(offsets),
            'velocities': torch.tensor(velocities),
            'pitches': torch.tensor(pitches),
        }

        if self.transform:
            mel_spectrogram = self.transform(mel_spectrogram)

        return mel_spectrogram, targets

    def pad_or_truncate(self, mel_spectrogram):
        current_length = mel_spectrogram.shape[-1]
        
        if current_length < self.target_length:
            # Pad with zeros (or another value)
            padding = self.target_length - current_length
            mel_spectrogram = torch.nn.functional.pad(mel_spectrogram, (0, padding))
        elif current_length > self.target_length:
            # Truncate to the target length
            mel_spectrogram = mel_spectrogram[:, :, :self.target_length]

        return mel_spectrogram
    
    def pad_or_truncate_target(self, target):
        # Ensure all targets are of the same length
        target_length = len(target)
        
        if target_length < self.target_length:
            # Pad with zeros (or another value)
            padding = self.target_length - target_length
            target = target + [0] * padding  # Pad with zero
        elif target_length > self.target_length:
            # Truncate to the target length
            target = target[:self.target_length]
        
        return target



# Example usage (replace with actual file paths)
#audio_paths = ['path_to_audio.wav']
#midi_paths = ['path_to_midi.mid']

maestro_CSV_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'dataset','maestro-v3.0.0', 'maestro-v3.0.0.csv'))
#print(f"maestro_CSV_path: {maestro_CSV_path}")
audio_paths, midi_paths = GetWavAndMidiPathFromcsv(maestro_CSV_path)

dataset = MAESTRODataset(audio_paths, midi_paths)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, input_channels=2, hidden_units=64, output_size=4):
        super(CRNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
        # Recurrent layers (LSTM)
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_units, batch_first=True)
        
        # Fully connected layers for regression
        self.fc = nn.Linear(hidden_units, output_size)  # output_size = 4 for onset, offset, frame, and velocity
    
    def forward(self, x):
        # Apply Conv Layers
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        
        # Reshape for LSTM (batch_size, seq_len, input_size)
        x = x.permute(0, 2, 3, 1)  # Now (batch_size, seq_len, features)
        x = x.reshape(x.size(0), x.size(1), -1)  # Flatten features for LSTM

        
        # Apply LSTM
        lstm_out, _ = self.lstm(x)
        
        # Use the last LSTM output for regression
        output = self.fc(lstm_out[:, -1, :])  # Get last timestep output
        
        return output

# Instantiate the model
model = CRNN(input_channels=2, hidden_units=128, output_size=4)  # 4 targets: onset, offset, frame, and velocity

import torch.optim as optim

# Loss function
criterion = nn.MSELoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    
    running_loss = 0.0
    for data in dataloader:
        mel_spectrograms, targets = data
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(mel_spectrograms)
        
        # Compute loss (one per target)
        loss = sum([criterion(outputs[:, i], targets[feature]) for i, feature in enumerate(targets)])
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(dataloader)}")
