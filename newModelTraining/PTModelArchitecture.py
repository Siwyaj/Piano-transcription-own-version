import torch
import torch.nn as nn
import torch.nn.functional as F

class OnsetOnlyCRNN(nn.Module):
    def __init__(self, n_bins, n_pitches, hidden_size=256):
        super().__init__()
        # Convolutional feature extractor (4 blocks like in the original model)
        self.conv1 = nn.Conv2d(1, 48, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(48, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 96, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(96, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(48)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(96)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool = nn.AvgPool2d((1, 2))  # Pool only in freq dimension

        # Calculate resulting feature size
        self.midfeat = (n_bins // (2**4)) * 128  # 4 pooling layers on freq axis

        self.fc5 = nn.Linear(self.midfeat, 768)
        self.bn5 = nn.BatchNorm1d(768)

        self.gru = nn.GRU(input_size=768, hidden_size=hidden_size, num_layers=2,
                          batch_first=True, bidirectional=True)

        self.fc_out = nn.Linear(hidden_size * 2, n_pitches)

    def forward(self, x):
        # x: (batch, 1, freq_bins, time_frames)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = F.dropout(x, p=0.2, training=self.training)

        # reshape for GRU
        batch_size, channels, freq_bins, time_steps = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(batch_size, time_steps, -1)  # (B, T, C*F)

        x = F.relu(self.bn5(self.fc5(x)))
        x = F.dropout(x, p=0.5, training=self.training)

        x, _ = self.gru(x)
        x = F.dropout(x, p=0.5, training=self.training)

        onset_pred = torch.sigmoid(self.fc_out(x))  # sigmoid for multilabel onset prediction
        return onset_pred
