import torch
import torch.nn as nn
import torch.nn.functional as F

class CRNNModel2(nn.Module):
    def __init__(self, mel_bins=229, num_classes=88, hidden_size=256):
        super(CRNNModel2, self).__init__()

        # Define the number of filters and kernel sizes for Conv1D layers
        self.conv1 = nn.Conv1d(in_channels=mel_bins, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        
        # Define GRU layers for temporal modeling
        self.gru1 = nn.GRU(input_size=256, hidden_size=hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        
        # Output layers for different types of predictions
        self.onset_fc = nn.Linear(hidden_size * 2, 1)  # Regression for onset (continuous)
        self.offset_fc = nn.Linear(hidden_size * 2, 1)  # Regression for offset (continuous)
        self.velocity_fc = nn.Linear(hidden_size * 2, 1)  # Regression for velocity (continuous)
        self.frame_fc = nn.Linear(hidden_size * 2, num_classes)  # Classification for frame-wise output (discrete)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """
        Args:
          x: (batch_size, time_steps, mel_bins)
        Outputs:
          output_dict: dict containing:
            'onset_output': (batch_size, time_steps, 1)  # Onset prediction
            'offset_output': (batch_size, time_steps, 1)  # Offset prediction
            'velocity_output': (batch_size, time_steps, 1)  # Velocity prediction
            'frame_output': (batch_size, time_steps, num_classes)  # Frame-wise classification
        """
        # Apply Conv1D layers
        x = x.transpose(1, 2)  # Change shape to (batch_size, mel_bins, time_steps)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.transpose(1, 2)  # Change back to (batch_size, time_steps, feature_size)

        # GRU layer for temporal modeling
        x, _ = self.gru1(x)

        # Apply Dropout
        x = self.dropout(x)

        # Output heads
        onset_output = self.onset_fc(x)  # Regression for onset (continuous)
        offset_output = self.offset_fc(x)  # Regression for offset (continuous)
        velocity_output = self.velocity_fc(x)  # Regression for velocity (continuous)
        frame_output = self.frame_fc(x)  # Classification for frame-wise output (discrete)

        # Return outputs as a dictionary
        output_dict = {
            'onset_output': onset_output,
            'offset_output': offset_output,
            'velocity_output': velocity_output,
            'frame_output': frame_output
        }

        return output_dict


# CUDA availability check
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CRNNModel2().to(device)
