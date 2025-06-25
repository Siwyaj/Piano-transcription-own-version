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

import torch
import torch.nn.functional as F

def onset_and_frames_loss(pitch_logits, frame_logits, pitch_labels, frame_labels, model=None):
    """
    pitch_logits, frame_logits: (batch_size, time_steps, 88)
    pitch_labels, frame_labels: same shape, binary or float targets
    model: torch.nn.Module, used to apply L2 regularization (optional)
    """

    # Ensure inputs are float32 and flattened
    pitch_logits = pitch_logits.reshape(-1, 88).float()
    frame_logits = frame_logits.reshape(-1, 88).float()
    pitch_labels = pitch_labels.reshape(-1, 88).float()
    frame_labels = frame_labels.reshape(-1, 88).float()

    # Binary labels for pitch (convert to 0/1)
    binary_pitch_labels = (pitch_labels > 0.0).float()

    pitch_labels_pos_weight = compute_batch_pos_weight(pitch_labels)
    frame_labels_pos_weight = compute_batch_pos_weight(frame_labels)

    # BCE with logits loss (default pos_weight = 1)
    pitch_loss = F.binary_cross_entropy_with_logits(pitch_logits, binary_pitch_labels, reduction='none', pos_weight=pitch_labels_pos_weight)
    frame_loss = F.binary_cross_entropy_with_logits(frame_logits, frame_labels, reduction='none', pos_weight=frame_labels_pos_weight)

    # Mean over pitch dimension (axis=2), sum over batch and time
    pitch_loss = pitch_loss.mean(dim=1).sum()
    frame_loss = frame_loss.mean(dim=1).sum()

    # L2 Regularization (if model is provided)
    l2_loss = 0.0
    if model is not None:
        for param in model.parameters():
            if param.requires_grad:
                l2_loss += torch.sum(param ** 2)
        l2_loss = 1e-4 * l2_loss  # Adjust weight as needed

    total_loss = pitch_loss + frame_loss + l2_loss

    return total_loss

def compute_batch_pos_weight(labels):
    pos = (labels == 1).sum().float()
    neg = (labels == 0).sum().float()
    if pos == 0:
        return torch.tensor(1.0).to(labels.device)  # avoid divide-by-zero
    return neg / (pos + 1e-6)

def train_onset_only(model, dataloader_train, dataloader_val, epochs=100, device='cuda'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    #criterion = nn.BCEWithLogitsLoss()
    model.train()

    for epoch in range(epochs):
        if epoch % 50 == 0:
            model_save_path = f'models/onset_only_model_total_epoch={epochs}_current_epoch={epoch}_n=10.pth'
            torch.save(model.state_dict(), model_save_path)
        total_loss = 0
        all_precisions, all_recalls, all_f1s = [], [], []
        for cqt, onsetLabel, sustainlabel in tqdm(dataloader_train):
            cqt = cqt.to(device)
            onsetLabel = onsetLabel.to(device)

            optimizer.zero_grad()

            # Forward pass
            logits = model(cqt)
            
            #pos_weight = compute_batch_pos_weight(onsetLabel).to(cqt.device)
            #loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss_fn = nn.BCEWithLogitsLoss()  # Use standard BCE loss
            loss = loss_fn(logits, onsetLabel.float())

            l2_loss = 0.0
            for param in model.parameters():
                if param.requires_grad:
                    l2_loss += torch.sum(param ** 2)
            l2_loss = 1e-4 * l2_loss
            loss += l2_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            with torch.no_grad():
                preds = torch.sigmoid(logits)
                #print(f"max: {preds.max().item()}, min: {preds.min().item()}, mean: {preds.mean().item()}")
                preds_bin = (preds > 0.5).float()
                y_true = onsetLabel.cpu().numpy().reshape(-1, 88)
                y_pred = preds_bin.cpu().numpy().reshape(-1, 88)
                for i in range(88):
                    if y_true[:, i].sum() > 0:
                        p = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
                        r = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
                        f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
                        all_precisions.append(p)
                        all_recalls.append(r)
                        all_f1s.append(f1)
        avg_loss = total_loss / len(dataloader_train)
        avg_precision = sum(all_precisions) / len(all_precisions) if all_precisions else 0
        avg_recall = sum(all_recalls) / len(all_recalls) if all_recalls else 0
        avg_f1 = sum(all_f1s) / len(all_f1s) if all_f1s else 0
        print("Training:")
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Precision: {avg_precision:.4f} | Recall: {avg_recall:.4f} | F1: {avg_f1:.4f}")
        write_to_log(f"[train] Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Precision: {avg_precision:.4f} | Recall: {avg_recall:.4f} | F1: {avg_f1:.4f}")


                # Validation
        model.eval()
        val_loss = 0
        val_precisions, val_recalls, val_f1s = [], [], []

        with torch.no_grad():
            for cqt, onsetLabel, sustainLabel in dataloader_val:
                cqt = cqt.to(device)
                onsetLabel = onsetLabel.to(device)

                logits = model(cqt)
                loss_fn = nn.BCEWithLogitsLoss()
                loss = loss_fn(logits, onsetLabel.float())

                # Add L2 regularization
                l2_loss = 0.0
                for param in model.parameters():
                    if param.requires_grad:
                        l2_loss += torch.sum(param ** 2)
                l2_loss = 1e-4 * l2_loss
                loss += l2_loss

                val_loss += loss.item()

                preds = torch.sigmoid(logits)
                preds_bin = (preds > 0.5).float()
                y_true = onsetLabel.cpu().numpy().reshape(-1, 88)
                y_pred = preds_bin.cpu().numpy().reshape(-1, 88)

                for i in range(88):
                    if y_true[:, i].sum() > 0:
                        p = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
                        r = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
                        f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
                        val_precisions.append(p)
                        val_recalls.append(r)
                        val_f1s.append(f1)

        avg_val_loss = val_loss / len(dataloader_val)
        avg_val_precision = sum(val_precisions) / len(val_precisions) if val_precisions else 0
        avg_val_recall = sum(val_recalls) / len(val_recalls) if val_recalls else 0
        avg_val_f1 = sum(val_f1s) / len(val_f1s) if val_f1s else 0

        print("Validation:")
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_val_loss:.4f} | Precision: {avg_val_precision:.4f} | Recall: {avg_val_recall:.4f} | F1: {avg_val_f1:.4f}")
        write_to_log(f"[VAL] Epoch {epoch+1}/{epochs} | Loss: {avg_val_loss:.4f} | Precision: {avg_val_precision:.4f} | Recall: {avg_val_recall:.4f} | F1: {avg_val_f1:.4f}")

        model.train()


        
def write_to_log(string):
    with open('log.txt', 'a') as f:
        f.write(string + '\n')
    #print(string)


if __name__ == "__main__":
    import config
    from torch.utils.data import DataLoader

    dataset_train = PianoDataset('HDF5/tolerance+-5_t_101_n_10train.hdf5')
    dataloader_train = DataLoader(dataset_train, batch_size=8, shuffle=True)
    dataset_val = PianoDataset('HDF5/tolerance+-5_t_101_n_2val.hdf5')
    dataloader_val = DataLoader(dataset_val, batch_size=8, shuffle=True)

    dataset_test = PianoDataset('HDF5/tolerance+-5_t_101_n_5test.hdf5')
    dataloader_test = DataLoader(dataset_test, batch_size=8, shuffle=True)
    '''
    
    
    cqt, onsetlabel, sustain = dataset[0]
    onsetration = []
    sustainratio = []
    for cqt, onsetlabel, sustainlabel in dataset:
        pos_count = (onsetlabel == 1).sum().item()
        neg_count = (onsetlabel == 0).sum().item()
        ratio = pos_count / (pos_count + neg_count)
        onsetration.append(ratio)
        pos_count = (sustainlabel == 1).sum().item()
        neg_count = (sustainlabel == 0).sum().item()
        ratio = pos_count / (pos_count + neg_count)
        sustainratio.append(ratio)
    ratio = sum(sustainratio) / len(sustainratio)
    print(f"sustainratio Positive ratio: {ratio:.4f}")
    ratio = sum(onsetration) / len(onsetration)
    print(f"onsetlabel Positive ratio: {ratio:.4f}")
    '''
    

    #model = OnsetOnlyCRNN(n_bins=config.n_bins, n_pitches=config.n_pitches)

    onsetModel = TAwareModel()

    #wholeModel = WholeModel()
    #model = WholeModel
    #train_joint(onsetModel, wholeModel, dataloader, epochs=epochs)

    epochs = 1000

    train_onset_only(onsetModel, dataloader_train, dataloader_val, epochs=epochs)

    #save models
    #sustain_model_save_path = f'models/sustain_model_epoch={epochs}_n=50.pth'
    pitch_model_save_path = f'models/before_exam_epochs={epochs}_n=10.pth'
    
    #torch.save(wholeModel.state_dict(), sustain_model_save_path)
    torch.save(onsetModel.state_dict(), pitch_model_save_path)
