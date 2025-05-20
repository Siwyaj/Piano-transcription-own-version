import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

class PianoTranscriptionModel(nn.Module):
    def __init__(self, n_bins=config.n_bins, n_pitches=config.n_pitches, hidden_size=128):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,3), padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3), padding=1)
        self.pool = nn.MaxPool2d((2,2))

        self.rnn_input_size = (n_bins // 4) * 64  # after pooling once on freq and time dim

        self.rnn = nn.GRU(input_size=self.rnn_input_size, hidden_size=hidden_size,
                          num_layers=2, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(hidden_size*2, n_pitches)

    def forward(self, x):
        # x: (batch, 1, freq_bins, time_frames)
        x = F.relu(self.conv1(x))
        x = self.pool(x)  # downsample freq and time by 2
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # downsample freq and time by 2 again
        # x shape: (batch, channels, freq_bins/4, time_frames/4)

        batch_size, channels, freq_bins, time_frames = x.shape
        # permute to (batch, time, channels*freq)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(batch_size, time_frames, channels * freq_bins)

        rnn_out, _ = self.rnn(x)
        out = self.fc(rnn_out)  # logits

        return out
