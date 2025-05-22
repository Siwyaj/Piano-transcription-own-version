import torch
import torch.nn as nn
print(torch.__version__)
class WholeModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Input channels = 2
        
        # Conv layers: (in_channels, out_channels, kernel_size, stride=1, padding=0)
        # We want output shapes matching described dimensions, so no padding.
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(5,3)) 
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(5,3), padding='same') 
        
        # Pool: 2x2 with stride (1,2)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,1))
        self.pool_pad = nn.ZeroPad2d((0, 1, 0, 0))

        self.conv3 = nn.Conv2d(16, 32, kernel_size=(5,3))
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(5,3))
        
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,1))
        
        self.conv5 = nn.Conv2d(64, 128, kernel_size=(5,3))
        
        # After conv5, reshape to (T, 10240)
        # Then linear layer to 1024
        self.fc1 = nn.Linear(80 * 128, 88)
        
        self.concat = nn.Linear(88, 176)
        # BiLSTM: input size 1024, hidden size 512, bidirectional
        self.bilstm = nn.LSTM(input_size=176, hidden_size=128, bidirectional=True, batch_first=True)
        
        self.attack_fc = nn.Linear(256, 88)

    def forward(self, x):
        # x shape: (batch, channels=2, time, width=356)
        #print(f"Input shape: {x.shape}")  # Debugging line
        
        x = self.conv1(x)
        #print(f"After conv1: {x.shape}")  # Debugging line
        x = self.conv2(x)
        #print(f"After conv2: {x.shape}")
        x = self.pool1(x)
        #print(f"After pool1: {x.shape}")
        x = self.pool_pad(x)  # Pad to (batch, 16, T, 179)
        #print(f"After pool1 and pool_pad: {x.shape}")

        x = self.conv3(x)
        #print(f"After conv3: {x.shape}")
        x = self.conv4(x)
        #print(f"After conv4: {x.shape}")
        x = self.pool2(x)
        x = self.pool_pad(x)
        #print(f"After pool2 and pool_pad: {x.shape}")

        x = self.conv5(x)
        #print(f"After conv5: {x.shape}")
        
        # x: (batch, channels=128, freq=80, time=101)

        x = x.permute(0, 3, 1, 2)  # → (batch, time=101, channels=128, freq=80)

        #print(f"After permute: {x.shape}")  # (8, 101, 128, 80)

        # Flatten last two dims: 128 * 80 = 10240
        x = x.reshape(x.size(0), x.size(1), -1)  # (batch, time, 10240)
        #print(f"After reshape: {x.shape}")

        
        # Dense layer per timestep
        x = self.fc1(x)  # (batch, T, 1024)
        #print(f"After fc: {x.shape}")
        
        x = self.concat(x)
        # BiLSTM expects input (batch, seq_len, input_size)
        x, _ = self.bilstm(x)  # output: (batch, T, 1024)
        #print(f"After LSTM: {x.shape}")

        # Final output shape: (batch, T, 1024)

        
        attack_logits = self.attack_fc(x)  # (batch, T, 88)
        #print(f"Final output shape: {attack_logits.shape}")
        
        return attack_logits

# Example usage:
# batch_size = 4, channels = 2, T+8=20, width=356
# input_tensor = torch.randn(4, 2, 20, 356)
# model = CustomModel()
# output = model(input_tensor)
# print(output.shape)  # should be (4, T, 1024)
