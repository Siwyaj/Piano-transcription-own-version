'''
This is the main script, it will contain all the functions that can run to do each part of the project.
this includes:
1. Data Preprocessing
2. training the model
3. testing the model
'''

import os
import sys
from DataLoader.GetWavAndMidiPathFromcsv import GetWavAndMidiPathFromcsv
import torch
import torch.nn as nn
import torch.optim as optim
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    # Check if each element in batch is a tuple with exactly two elements
    if len(batch) > 0:
        # Check the first element of the batch to see if it's a tuple with two items
        if isinstance(batch[0], tuple) and len(batch[0]) == 2:
            inputs, targets = zip(*batch)
        else:
            raise ValueError(f"Expected tuple (input, target) but got {type(batch[0])} with length {len(batch[0])}")
    
    # Pad the sequences
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=0)
    
    return padded_inputs, padded_targets


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    maestro_CSV_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'dataset','maestro-v3.0.0', 'maestro-v3.0.0.csv'))
    print(f"maestro_CSV_path: {maestro_CSV_path}")
    wav_paths, midi_paths = GetWavAndMidiPathFromcsv(maestro_CSV_path)
    print(f"wav_paths: {wav_paths[0]}")
    print(f"midi_paths: {midi_paths[0]}")

    from DataLoader.MaestroDatasetLoader import MaestroDataset
    from torch.utils.data import DataLoader

    # Instantiate dataset
    dataset = MaestroDataset(wav_paths, midi_paths)

    # Wrap in DataLoader
    train_loader = DataLoader(
        dataset,
        batch_size=16,         # Load 16 samples at a time
        shuffle=True,          # Shuffle every epoch
        num_workers=8,         # Load data in parallel (adjust based on your CPU)
        collate_fn=collate_fn, # Use the custom collate function
    )



    from models.NewModel import CRNNModel2
    model = CRNNModel2().to(device)
    criterion_regression = nn.MSELoss()  # For regression tasks like onset, offset, velocity
    criterion_classification = nn.CrossEntropyLoss()  # For frame-wise classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10  # Number of epochs to train
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        for batch_idx, data in enumerate(train_loader):
            print(f"Batch {batch_idx+1}/{len(train_loader)}")
            inputs = data['input'].to(device)  # Input (spectrogram or log-mel)
            labels = data['labels']  # Dictionary containing labels

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            output_dict = model(inputs)

            # Calculate loss for each output
            loss_onset = criterion_regression(output_dict['onset_output'], labels['onset'])
            loss_offset = criterion_regression(output_dict['offset_output'], labels['offset'])
            loss_velocity = criterion_regression(output_dict['velocity_output'], labels['velocity'])
            loss_frame = criterion_classification(output_dict['frame_output'], labels['frame'])

            # Total loss
            total_loss = loss_onset + loss_offset + loss_velocity + loss_frame

            # Backward pass and optimize
            total_loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {total_loss.item():.4f}")

if __name__ == "__main__":
    print("Starting training...")
    train()
    pass