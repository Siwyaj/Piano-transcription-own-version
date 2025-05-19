import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Define your CRNNModel2 (mel_bins=256)
import config
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, momentum):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum)
        self.init_weight()

    def forward(self, x, pool_size=(2,2)):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu_(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu_(x)
        x = F.avg_pool2d(x, kernel_size=pool_size)
        return x

    def init_weight(self):
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        self.bn1.weight.data.fill_(1.)
        self.bn1.bias.data.fill_(0.)
        self.bn2.weight.data.fill_(1.)
        self.bn2.bias.data.fill_(0.)

class CRNNModelMultiHead(nn.Module):
    def __init__(self, mel_bins=config.mel_bins, num_classes=config.num_classes, momentum=0.1):
        super(CRNNModelMultiHead, self).__init__()
        
        # Conv layers expect input (batch, 1, time, freq)
        self.convBlock1 = ConvBlock(1, 48, momentum)
        self.convBlock2 = ConvBlock(48, 64, momentum)
        self.convBlock3 = ConvBlock(64, 96, momentum)
        self.convBlock4 = ConvBlock(96, 128, momentum)
        
        # Calculate midfeat after convs and pooling on freq dimension:
        # pooling (1,2) applied 4 times => freq_bins // 2^4 = 229//16 = 14 (integer div)
        midfeat = 128 * 14  # channels * freq_bins reduced
        
        self.fc5 = nn.Linear(midfeat, 768, bias=False)
        self.bn5 = nn.BatchNorm1d(768, momentum)
        
        self.gru = nn.GRU(input_size=768, hidden_size=256, num_layers=2, 
                          batch_first=True, bidirectional=True)
        
        # Four separate heads for onset, offset, velocity, frame
        self.onset_fc = nn.Linear(512, num_classes)
        self.offset_fc = nn.Linear(512, num_classes)
        self.velocity_fc = nn.Linear(512, num_classes)
        self.frame_fc = nn.Linear(512, num_classes)
        
        self.dropout = nn.Dropout(0.5)
        
        self.init_weight()
    
    def init_weight(self):
        nn.init.xavier_uniform_(self.fc5.weight)
        self.bn5.weight.data.fill_(1.)
        self.bn5.bias.data.fill_(0.)

        nn.init.xavier_uniform_(self.onset_fc.weight)
        self.onset_fc.bias.data.fill_(0.)
        nn.init.xavier_uniform_(self.offset_fc.weight)
        self.offset_fc.bias.data.fill_(0.)
        nn.init.xavier_uniform_(self.velocity_fc.weight)
        self.velocity_fc.bias.data.fill_(0.)
        nn.init.xavier_uniform_(self.frame_fc.weight)
        self.frame_fc.bias.data.fill_(0.)

    def forward(self, x):
        # Input x: (batch, time, mel_bins)
        # Reshape to (batch, 1, time, freq) to use Conv2d
        x = x.unsqueeze(1)
        
        x = self.convBlock1(x, pool_size=(1,2))  # pooling only freq dim
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.convBlock2(x, pool_size=(1,2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.convBlock3(x, pool_size=(1,2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.convBlock4(x, pool_size=(1,2))
        x = F.dropout(x, p=0.2, training=self.training)
        
        # x shape now: (batch, 128, time, 14)
        x = x.transpose(1, 2).flatten(2)  # (batch, time, 128*14=1792)
        
        # fc5 + batchnorm + relu
        x = self.fc5(x)  # (batch, time, 768)
        # batchnorm1d expects (batch*time, features), so reshape
        b, t, f = x.shape
        x = self.bn5(x.reshape(b*t, f)).reshape(b, t, f)
        x = F.relu(x)
        x = self.dropout(x)
        
        # GRU
        x, _ = self.gru(x)  # (batch, time, 512)
        x = self.dropout(x)
        
        # 4 heads with sigmoid activation
        onset_output = torch.sigmoid(self.onset_fc(x))
        offset_output = torch.sigmoid(self.offset_fc(x))
        velocity_output = torch.sigmoid(self.velocity_fc(x))
        frame_output = torch.sigmoid(self.frame_fc(x))
        
        return {
            'onset_output': onset_output,
            'offset_output': offset_output,
            'velocity_output': velocity_output,
            'frame_output': frame_output
        }



# Focal Loss for binary classification (onsets etc)
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        modulating_factor = (1 - p_t) ** self.gamma
        loss = alpha_factor * modulating_factor * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# Compute F1 score (micro) for onset detection (batch_size, time, 1)
def f1_score(outputs, targets, threshold=0.5, eps=1e-7):
    outputs = torch.sigmoid(outputs)
    outputs = (outputs > threshold).float()
    targets = targets.float()

    tp = (outputs * targets).sum()
    fp = (outputs * (1 - targets)).sum()
    fn = ((1 - outputs) * targets).sum()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    return f1.item()


# Compute accuracy for frame outputs (batch, time, classes)
def accuracy(outputs, targets):
    preds = torch.argmax(outputs, dim=2)
    targets = torch.argmax(targets, dim=2)
    correct = (preds == targets).float()
    return correct.mean().item()

def bce(output, target, mask):
    eps = 1e-7
    output = torch.clamp(output, eps, 1. - eps)
    matrix = - target * torch.log(output) - (1. - target) * torch.log(1. - output)
    return torch.sum(matrix * mask) / torch.sum(mask)

# Your loss calculation combining heads
import torch.nn.functional as F

def compute_loss(output_dict, target_dict):
    # Onset and offset: binary classification (use BCE with logits)
    pos_weight = torch.tensor([10.0], device=output_dict['onset_output'].device)  # Tune 10.0 as needed
    onset_loss = F.binary_cross_entropy_with_logits(
        output_dict['onset_output'], target_dict['onset'], pos_weight=pos_weight
    )

    
    offset_loss = F.binary_cross_entropy_with_logits(
        output_dict['offset_output'], target_dict['offset']
    )

    # Velocity: regression (normalize to 0–1, use MSE)
    velocity_loss = F.mse_loss(
        torch.sigmoid(output_dict['velocity_output']),  # optional sigmoid if targets are [0,1]
        target_dict['velocity'] / 128.0
    )

    # Frame: multi-label classification → use BCE with logits
    frame_loss = F.binary_cross_entropy_with_logits(
        output_dict['frame_output'], target_dict['frame']
    )

    # Total loss
    total_loss = onset_loss + offset_loss + velocity_loss + frame_loss

    # Return both total and individual losses
    return total_loss, {
        'onset': onset_loss.item(),
        'offset': offset_loss.item(),
        'velocity': velocity_loss.item(),
        'frame': frame_loss.item()
    }


import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Apply sigmoid to get probabilities
        inputs = torch.clamp(inputs, 1e-7, 1 - 1e-7)
        inputs = torch.sigmoid(inputs)

        # BCE loss
        bce = -targets * torch.log(inputs) - (1 - targets) * torch.log(1 - inputs)

        # Focal factor
        pt = inputs * targets + (1 - inputs) * (1 - targets)
        focal_weight = (1 - pt) ** self.gamma
        loss = self.alpha * focal_weight * bce

        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()



def train(model, dataloader, optimizer, device, epochs=30):
    model.train()
    #focal_loss = FocalLoss()
    for epoch in range(epochs):
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for mel, targets in progress:
            mel = mel.to(device)           # (batch, time, mel_bins)
            targets = {k: v.to(device) for k, v in targets.items()}

            optimizer.zero_grad()
            outputs = model(mel)

            loss, loss_dict = compute_loss(outputs, targets)

            loss.backward()
            optimizer.step()

            # Calculate metrics on this batch
            batch_acc = accuracy(outputs['frame_output'], targets['frame'])
            batch_f1 = f1_score(outputs['onset_output'], targets['onset'])

            progress.set_postfix({
            'Loss': f"{loss.item():.4f}",
            'Acc': f"{batch_acc * 100:.2f}%",
            'F1_onset': f"{batch_f1:.4f}",
            'Onset': f"{loss_dict['onset']:.4f}",
            'Offset': f"{loss_dict['offset']:.4f}",
            'Velocity': f"{loss_dict['velocity']:.4f}",
            'Frame': f"{loss_dict['frame']:.4f}"
            })



    # Load your dataset here
    
    
    #dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)


def train_onset(model, dataloader, optimizer, device, epochs=5):
    model.train()
    
    #worked with 5 epochs
    bce = nn.BCELoss()

    #added folcal loss
    #focal_loss = FocalLoss(alpha=0.25, gamma=2.0)



    for epoch in range(epochs):
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for mel, labels in progress:
            mel = mel.to(device)
            onset_target = labels["onset"].to(device)

            optimizer.zero_grad()
            onset_pred = model(mel)
            
            #5 epochs
            loss = bce(onset_pred, onset_target)
            
            #added focal loss
            #loss = focal_loss(onset_pred, onset_target)

            loss.backward()
            optimizer.step()

            # Optionally calculate a simple F1 or accuracy
            with torch.no_grad():
                pred_bin = (onset_pred > 0.1).float()
                tp = (pred_bin * onset_target).sum()
                fp = (pred_bin * (1 - onset_target)).sum()
                fn = ((1 - pred_bin) * onset_target).sum()
                f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)

            progress.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'F1': f"{f1.item():.4f}"
            })


if __name__ == "__main__":
    # Example usage
    from models.NewModel import CRNNModel2
    from DataLoader.PrepareDataFromHdf5 import DataLoaderHdf5
    from models.stoldenArchitecture import MultiTaskCRNN
    from models.OnlyOnset import OnsetOnlyCRNN

    # Initialize model, optimizer, and dataloader
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = OnsetOnlyCRNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    dataloader = DataLoaderHdf5("hdf5Files/train_hdf5_file.h5", max_samples=None)
    for mel, targets in dataloader:
        print("Onset mean:", targets["onset"].mean().item())
        print("Offset mean:", targets["offset"].mean().item())
        print("Velocity mean:", targets["velocity"].mean().item())
        break

    # Train the model
    #train(model, dataloader, optimizer, device=device)
    train_onset(model, dataloader, optimizer, device=device, epochs=75)
    torch.save(model.state_dict(), "crnn_model_final.pth")
    #TO DO
    #add validation to model
    #add test to model
    #transfer model architecture to new script