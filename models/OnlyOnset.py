import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

class OnsetOnlyCRNN(nn.Module):
    def __init__(self, mel_bins=config.mel_bins, num_classes=config.num_classes, hidden_size=256):
        super(OnsetOnlyCRNN, self).__init__()

        self.conv1 = nn.Conv1d(mel_bins, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)

        self.gru = nn.GRU(256, hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):  # x: (B, T, mel_bins)
        x = x.transpose(1, 2)  # (B, mel_bins, T)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.transpose(1, 2)  # (B, T, features)

        x, _ = self.gru(x)
        x = self.dropout(x)
        onset_out = torch.sigmoid(self.fc(x))  # (B, T, 88)
        return onset_out
