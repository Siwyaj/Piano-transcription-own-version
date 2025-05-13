'''
This script contains the CRNN model class and the training function.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from CRNNsubmodel import CRNNSubModel
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

class CRNNModel(nn.Module):
    def __init__(self):
        super(CRNNModel, self).__init__()
        classes_num = config.classes_num
        sample_rate = config.sample_rate
        window_size = config.window_size
        hop_size = sample_rate // config.frames_per_second
        mel_bins = config.mel_bins
        fmin = 30
        fmax = sample_rate // 2

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        midfeat = 1792
        momentum = 0.01
        
        self.batchNorm0 = nn.BatchNorm2d(mel_bins, momentum)

        self.frame_model = CRNNSubModel(classes_num, midfeat, momentum)
        self.reg_onset_model = CRNNSubModel(classes_num, midfeat, momentum)
        self.reg_offset_model = CRNNSubModel(classes_num, midfeat, momentum)
        self.velocity_model = CRNNSubModel(classes_num, midfeat, momentum)

        self.reg_onset_gru = nn.GRU(input_size=88 * 2, hidden_size=256, num_layers=1, 
            bias=True, batch_first=True, dropout=0., bidirectional=True)
        self.reg_onset_fc = nn.Linear(512, classes_num, bias=True)

        self.frame_gru = nn.GRU(input_size=88 * 3, hidden_size=256, num_layers=1, 
            bias=True, batch_first=True, dropout=0., bidirectional=True)
        self.frame_fc = nn.Linear(512, classes_num, bias=True)

        self.init_weight()
        
    def init_weight(self):
        import math

        # ---- Initialize BatchNorm (self.bn0) ----
        self.bn0.weight.data.fill_(1.)
        self.bn0.bias.data.fill_(0.)

        # ---- GRU Initializer Helpers ----
        def _concat_init(tensor, init_funcs):
            length, fan_out = tensor.shape
            fan_in = length // len(init_funcs)
            for i, init_func in enumerate(init_funcs):
                init_func(tensor[i * fan_in : (i + 1) * fan_in, :])

        def _inner_uniform(tensor):
            fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
            bound = math.sqrt(3.0 / fan_in)
            nn.init.uniform_(tensor, -bound, bound)

        # ---- Initialize reg_onset_gru ----
        for i in range(self.reg_onset_gru.num_layers):
            _concat_init(getattr(self.reg_onset_gru, f'weight_ih_l{i}'),
                        [_inner_uniform, _inner_uniform, _inner_uniform])
            getattr(self.reg_onset_gru, f'bias_ih_l{i}').data.fill_(0.)

            _concat_init(getattr(self.reg_onset_gru, f'weight_hh_l{i}'),
                        [_inner_uniform, _inner_uniform, nn.init.orthogonal_])
            getattr(self.reg_onset_gru, f'bias_hh_l{i}').data.fill_(0.)

        # ---- Initialize frame_gru ----
        for i in range(self.frame_gru.num_layers):
            _concat_init(getattr(self.frame_gru, f'weight_ih_l{i}'),
                        [_inner_uniform, _inner_uniform, _inner_uniform])
            getattr(self.frame_gru, f'bias_ih_l{i}').data.fill_(0.)

            _concat_init(getattr(self.frame_gru, f'weight_hh_l{i}'),
                        [_inner_uniform, _inner_uniform, nn.init.orthogonal_])
            getattr(self.frame_gru, f'bias_hh_l{i}').data.fill_(0.)

        # ---- Initialize reg_onset_fc ----
        nn.init.xavier_uniform_(self.reg_onset_fc.weight)
        if self.reg_onset_fc.bias is not None:
            self.reg_onset_fc.bias.data.fill_(0.)

        # ---- Initialize frame_fc ----
        nn.init.xavier_uniform_(self.frame_fc.weight)
        if self.frame_fc.bias is not None:
            self.frame_fc.bias.data.fill_(0.)

    def forward(self, input):
        """
        Args:
          input: (batch_size, data_length)

        Outputs:
          output_dict: dict, {
            'reg_onset_output': (batch_size, time_steps, classes_num),
            'reg_offset_output': (batch_size, time_steps, classes_num),
            'frame_output': (batch_size, time_steps, classes_num),
            'velocity_output': (batch_size, time_steps, classes_num)
          }
        """

        x = self.spectrogram_extractor(input)
        x = self.logmel_extractor(x)

        x = x.transpose(1, 3)
        x = self.batchNorm0(x)
        x = x.transpose(1, 3)

        frame_output = self.frame_model(x)
        reg_onset_output = self.reg_onset_model(x)
        reg_offset_output = self.reg_offset_model(x)
        velocity_output = self.velocity_model(x)

        # Use velocities to condition onset regression
        x = torch.cat((reg_onset_output, (reg_onset_output ** 0.5) * velocity_output.detach()), dim=2)
        (x, _) = self.reg_onset_gru(x)
        x = F.dropout(x, p=0.5, training=self.training, inplace=False)
        reg_onset_output = torch.sigmoid(self.reg_onset_fc(x))
        """(batch_size, time_steps, classes_num)"""

        # Use onsets and offsets to condition frame-wise classification
        x = torch.cat((frame_output, reg_onset_output.detach(), reg_offset_output.detach()), dim=2)
        (x, _) = self.frame_gru(x)
        x = F.dropout(x, p=0.5, training=self.training, inplace=False)
        frame_output = torch.sigmoid(self.frame_fc(x))  # (batch_size, time_steps, classes_num)
        """(batch_size, time_steps, classes_num)"""

        output_dict = {
            'reg_onset_output': reg_onset_output, 
            'reg_offset_output': reg_offset_output, 
            'frame_output': frame_output, 
            'velocity_output': velocity_output}

        return output_dict
