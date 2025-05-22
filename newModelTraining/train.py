import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
from newModelTraining.model import PianoTranscriptionModel
from newModelTraining.PTModelArchitecture import OnsetOnlyCRNN
from newModelTraining.prepareData import PianoDataset
from newModelTraining.tawareModel import TAwareModel
from newModelTraining.WholeModel import WholeModel

from sklearn.metrics import precision_score, recall_score, f1_score

import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Apply sigmoid if inputs are raw logits
        inputs = torch.sigmoid(inputs)

        # Flatten tensors to simplify calculation
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)


        intersection = (inputs * targets).sum()
        dice_coeff = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice_coeff

import torch
import torch.nn as nn

def joint_loss(attack_logits, whole_logits, attack_labels, whole_labels):
    # Initialize binary cross-entropy with logits loss
    bce_loss = nn.BCEWithLogitsLoss()

    # Calculate losses
    lattack = bce_loss(attack_logits, attack_labels.float())
    lwhole = bce_loss(whole_logits, whole_labels.float())

    # Sum the losses for joint training
    lnote = lattack + lwhole

    return lnote


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

criterion = FocalLoss(alpha=1, gamma=2)


def evaluate_batch(preds, labels):
    preds_bin = (preds > 0.4).float()
    y_true = labels.cpu().numpy().flatten()
    y_pred = preds_bin.cpu().numpy().flatten()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return precision, recall, f1

import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

def train(onsetModel, wholeModel, dataloader, epochs=100, device='cuda'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        all_precisions = []
        all_recalls = []
        all_f1s = []
        

        for cqt, label in dataloader:
            cqt = cqt.to(device)
            label = label.to(device)
            if label.sum() == 0:
                continue  # Skip batches with no positive labels

            label_down = label[:, ::4, :]      # Downsample time axis
            label_down = label_down[:, 4:-3, :]  # Crop time axis
            #print("Label shape:", label_down.shape)

            #print("Label sum:", label.sum().item(), "Label_down sum:", label_down.sum().item())

            preds = model(cqt)  # Output shape: (batch, T, 88)
            loss = criterion(preds, label_down.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            #print("Preds min/max:", preds.min().item(), preds.max().item())

            # --- Compute metrics ---
            with torch.no_grad():
                pred_binary = (torch.sigmoid(preds) > 0.2).float()
                label_flat = label_down.cpu().numpy().reshape(-1, 88)
                pred_flat = pred_binary.cpu().numpy().reshape(-1, 88)
                for i in range(88):  # per-pitch average (optional, or do macro avg)
                    if label_flat[:, i].sum() > 0:
                        p = precision_score(label_flat[:, i], pred_flat[:, i], zero_division=0)
                        r = recall_score(label_flat[:, i], pred_flat[:, i], zero_division=0)
                        f1 = f1_score(label_flat[:, i], pred_flat[:, i], zero_division=0)

                        all_precisions.append(p)
                        all_recalls.append(r)
                        all_f1s.append(f1)
        
        avg_loss = total_loss / len(dataloader)
        avg_precision = sum(all_precisions) / len(all_precisions) if all_precisions else 0
        avg_recall = sum(all_recalls) / len(all_recalls) if all_recalls else 0
        avg_f1 = sum(all_f1s) / len(all_f1s) if all_f1s else 0

        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Precision: {avg_precision:.4f} | Recall: {avg_recall:.4f} | F1: {avg_f1:.4f}")






if __name__ == "__main__":
    import config
    from torch.utils.data import DataLoader

    dataset = PianoDataset('HDF5/onetrain.hdf5')
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    cqt, label = dataset[0]
    ratios = []
    
    for cqt, label in dataset:
        pos_count = (label == 1).sum().item()
        neg_count = (label == 0).sum().item()
        ratio = pos_count / (pos_count + neg_count)
        ratios.append(ratio)
    ratio = sum(ratios) / len(ratios)
    print(f"Positive ratio: {ratio:.4f}")
    print("Sample label sum:", label.sum().item())  # should be > 0 if there are notes
    print("Label shape:", label.shape)

    #model = OnsetOnlyCRNN(n_bins=config.n_bins, n_pitches=config.n_pitches)
    onsetModel = TAwareModel()
    wholeModel = WholeModel()
    model = WholeModel
    train(onsetModel, wholeModel, dataloader, epochs=1000)
