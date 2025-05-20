import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os
import torch
import torch.nn as nn
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from newModelTraining.model import PianoTranscriptionModel
from newModelTraining.prepareData import PianoDataset

from sklearn.metrics import precision_score, recall_score, f1_score

import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score


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
        inputs = inputs.view(-1)
        targets = targets.reshape(-1)


        intersection = (inputs * targets).sum()
        dice_coeff = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice_coeff



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

def train(model, dataloader, epochs=100, device='cuda'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Use your FocalLoss here, adjust alpha as needed (0 < alpha < 1)
    #criterion = FocalLoss(alpha=0.9, gamma=2, logits=True)
    criterion = DiceLoss()

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
                continue  # skip batch with no positives

            #print("Original label sum:", label.sum().item())
            label_down = label[:, ::4, :]
            #print("Downsampled label sum:", label_down.sum().item())

            preds = model(cqt)
            #print("Preds shape:", preds.shape)
            #print("Label_down shape:", label_down.shape)


            loss = criterion(preds, label_down)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Compute metrics
            precision, recall, f1 = evaluate_batch(torch.sigmoid(preds), label_down)
            all_precisions.append(precision)
            all_recalls.append(recall)
            all_f1s.append(f1)
            #print('Preds shape:', preds.shape)
            #print('Label shape:', label_down.shape)
            #print('Label sum:', label_down.sum().item())
            #print('Preds sample (sigmoid):', torch.sigmoid(preds[0, :5, 0]))
            #print('Original label sum:', label.sum().item())

        avg_loss = total_loss / len(dataloader)
        avg_precision = sum(all_precisions) / len(all_precisions)
        avg_recall = sum(all_recalls) / len(all_recalls)
        avg_f1 = sum(all_f1s) / len(all_f1s)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} | Precision: {avg_precision:.4f} | Recall: {avg_recall:.4f} | F1: {avg_f1:.4f}")




if __name__ == "__main__":
    import config
    from torch.utils.data import DataLoader

    dataset = PianoDataset('HDF5/train.hdf5')
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

    model = PianoTranscriptionModel()
    train(model, dataloader, epochs=1000)
