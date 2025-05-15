'''
This script holds the training function used in main.py, to train the model
It uses the model in models/newModel.py and load the data from hdf5Files/train_hdf5_file using Dataloader/PrepareDataFromHdf5.py
'''
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Instantiate loss functions
mse = torch.nn.MSELoss()
bce = torch.nn.BCEWithLogitsLoss()  # good for frame_output (multi-label)
from tqdm import tqdm
import torch.nn.functional as F

def save_model(model, path, model_name):
    """
    Save the model state dictionary to the specified path.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path, model_name)
    if os.path.exists(path):
        print(f"Model {model_name} already exists. Overwriting...")
    
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def compute_loss(outputs, targets):
    """
    Compute losses for each output.
    Adjust this based on your actual loss strategy.
    """
    loss_onset = F.mse_loss(outputs['onset_output'], targets['onset'])  # Regression
    loss_offset = F.mse_loss(outputs['offset_output'], targets['offset'])  # Regression
    loss_velocity = F.mse_loss(outputs['velocity_output'], targets['velocity'])  # Regression
    loss_frame = F.binary_cross_entropy_with_logits(outputs['frame_output'], targets['frame'])  # Classification

    total_loss = loss_onset + loss_offset + loss_velocity + loss_frame

    return total_loss, {
        'loss_onset': loss_onset.item(),
        'loss_offset': loss_offset.item(),
        'loss_velocity': loss_velocity.item(),
        'loss_frame': loss_frame.item(),
    }

def calculate_accuracy(preds, targets):
    """
    Binary accuracy for frame prediction (thresholded at 0.5).
    Assumes inputs are logits.
    """
    with torch.no_grad():
        preds_binary = (torch.sigmoid(preds) > 0.5).float()
        correct = (preds_binary == targets).float().sum()
        total = targets.numel()
        return (correct / total).item()

def train(model, dataloader, optimizer, num_epochs=10, device='cuda'):
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0

        progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for mel, targets in progress:
            mel = mel.to(device)  # (B, T, 256)
            for k in targets:
                targets[k] = targets[k].to(device)

            optimizer.zero_grad()
            outputs = model(mel)
            loss, loss_dict = compute_loss(outputs, targets)
            loss.backward()
            optimizer.step()

            # Accuracy for frame prediction
            batch_acc = calculate_accuracy(outputs['frame_output'], targets['frame'])

            # Logging
            epoch_loss += loss.item()
            epoch_acc += batch_acc

            progress.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Acc': f"{batch_acc*100:.2f}%",
                'Onset': f"{loss_dict['loss_onset']:.4f}",
                'Frame': f"{loss_dict['loss_frame']:.4f}"
            })

        avg_loss = epoch_loss / len(dataloader)
        avg_acc = epoch_acc / len(dataloader)
        print(f"Epoch {epoch+1} done. Avg Loss: {avg_loss:.4f}, Avg Frame Acc: {avg_acc*100:.2f}%\n")
    
    #save model
    save_model(model, "models/saved_models", f"initial_attempt_acc={avg_acc*100:.2f}_loss={avg_loss:.4f}_{epoch+1}.pth")


if __name__ == "__main__":
    # Example usage
    from models.NewModel import CRNNModel2
    from DataLoader.PrepareDataFromHdf5 import DataLoaderHdf5

    # Initialize model, optimizer, and dataloader
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CRNNModel2().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    dataloader = DataLoaderHdf5("hdf5Files/train_hdf5_file")

    # Train the model
    train(model, dataloader, optimizer, num_epochs=10, device=device)