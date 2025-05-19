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

    def forward(self, x, pool_size=(1, 2)):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x, kernel_size=pool_size)
        return x

    def init_weight(self):
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        self.bn1.weight.data.fill_(1.)
        self.bn1.bias.data.fill_(0.)
        self.bn2.weight.data.fill_(1.)
        self.bn2.bias.data.fill_(0.)


class AcousticSubModel(nn.Module):
    def __init__(self, classes_num, midfeat, momentum):
        super(AcousticSubModel, self).__init__()
        self.conv1 = ConvBlock(1, 48, momentum)
        self.conv2 = ConvBlock(48, 64, momentum)
        self.conv3 = ConvBlock(64, 96, momentum)
        self.conv4 = ConvBlock(96, 128, momentum)

        self.fc = nn.Linear(midfeat, 768, bias=False)
        self.bn = nn.BatchNorm1d(768, momentum)
        self.gru = nn.GRU(768, 256, num_layers=2, batch_first=True, bidirectional=True)
        self.output_fc = nn.Linear(512, classes_num)

    def forward(self, x):
        x = self.conv1(x)
        x = F.dropout(x, 0.2, self.training)
        x = self.conv2(x)
        x = F.dropout(x, 0.2, self.training)
        x = self.conv3(x)
        x = F.dropout(x, 0.2, self.training)
        x = self.conv4(x)
        x = F.dropout(x, 0.2, self.training)

        x = x.transpose(1, 2).flatten(2)  # (B, T, C*F)
        b, t, f = x.shape
        x = self.bn(self.fc(x).reshape(b * t, -1)).reshape(b, t, -1)
        x = F.relu(x)
        x = F.dropout(x, 0.5, self.training)
        x, _ = self.gru(x)
        x = F.dropout(x, 0.5, self.training)
        return torch.sigmoid(self.output_fc(x))


class MultiTaskCRNN(nn.Module):
    def __init__(self, mel_bins=256, classes_num=88, momentum=0.01):
        super(MultiTaskCRNN, self).__init__()
        midfeat = 128 * (mel_bins // 16)  # 4 poolings
        self.onset_model = AcousticSubModel(classes_num, midfeat, momentum)
        self.offset_model = AcousticSubModel(classes_num, midfeat, momentum)
        self.velocity_model = AcousticSubModel(classes_num, midfeat, momentum)
        self.frame_model = AcousticSubModel(classes_num, midfeat, momentum)

        self.onset_gru = nn.GRU(88 * 2, 256, num_layers=1, batch_first=True, bidirectional=True)
        self.onset_fc = nn.Linear(512, classes_num)

        self.frame_gru = nn.GRU(88 * 3, 256, num_layers=1, batch_first=True, bidirectional=True)
        self.frame_fc = nn.Linear(512, classes_num)

    def forward(self, mel):
        x = mel.unsqueeze(1)  # (B, 1, T, F)

        onset = self.onset_model(x)      # (B, T, 88)
        offset = self.offset_model(x)
        velocity = self.velocity_model(x)
        frame = self.frame_model(x)

        # Onset conditioning
        onset_feat = torch.cat([onset, (onset ** 0.5) * velocity.detach()], dim=-1)
        onset_feat, _ = self.onset_gru(onset_feat)
        onset_out = torch.sigmoid(self.onset_fc(onset_feat))

        # Frame conditioning
        frame_feat = torch.cat([frame, onset_out.detach(), offset.detach()], dim=-1)
        frame_feat, _ = self.frame_gru(frame_feat)
        frame_out = torch.sigmoid(self.frame_fc(frame_feat))

        return {
            'onset_output': onset_out,
            'offset_output': offset,
            'velocity_output': velocity,
            'frame_output': frame_out
        }
