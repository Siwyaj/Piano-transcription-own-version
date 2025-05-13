'''
This script contains the CRNN model which will be used in CRNNsupermodel.py.
This model will have 4 instances in the CRNNsupermodel.py script.
each for:
    -note onset
    -note offset
    -velocity
    -frame
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, momentum):
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.batchNorm1 = nn.BatchNorm2d(out_channels, momentum)
        self.batchNorm2 = nn.BatchNorm2d(out_channels, momentum)

        self.init_weight()

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        """
        Args:
          input: (batch_size, in_channels, time_steps, freq_bins)

        Outputs:
          output: (batch_size, out_channels, classes_num)
        """
        x = self.conv1(input)
        x = self.batchNorm1(x)
        x = F.relu_(x)
        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = F.relu_(x)
        x = F.avg_pool2d(x, kernel_size=pool_size)
        
        return x
    
    def init_weight(self):
        # Initialize conv1 weights using Xavier uniform
        nn.init.xavier_uniform_(self.conv1.weight)
        # No bias to initialize since bias=False in conv1

        # Initialize conv2 weights using Xavier uniform
        nn.init.xavier_uniform_(self.conv2.weight)
        # No bias to initialize since bias=False in conv2

        # Initialize BatchNorm1 parameters: weight (gamma) = 1, bias (beta) = 0
        self.batchNorm1.weight.data.fill_(1.)
        self.batchNorm1.bias.data.fill_(0.)

        # Initialize BatchNorm2 parameters: weight (gamma) = 1, bias (beta) = 0
        self.batchNorm2.weight.data.fill_(1.)
        self.batchNorm2.bias.data.fill_(0.)



class CRNNSubModel(nn.Module):
    def __init__(self, classes_num, midfeat, momentum):
        super(CRNNSubModel, self).__init__()
        
        self.convBlock1 = ConvBlock(in_channels=1, out_channels=48, momentum=momentum)
        self.convBlock2 = ConvBlock(in_channels=48, out_channels=64, momentum=momentum)
        self.convBlock3 = ConvBlock(in_channels=64, out_channels=96, momentum=momentum)
        self.convBlock4 = ConvBlock(in_channels=96, out_channels=128, momentum=momentum)
        
        self.fc5 = nn.Linear(midfeat, 768, bias=False)
        self.batchnorm5 = nn.BatchNorm1d(768, momentum=momentum)

        self.gru = nn.GRU(input_size=768, hidden_size=256, num_layers=2, 
            bias=True, batch_first=True, dropout=0., bidirectional=True)

        self.fc = nn.Linear(512, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        import math

        # ----- Initialize fc5 -----
        nn.init.xavier_uniform_(self.fc5.weight)
        if self.fc5.bias is not None:
            self.fc5.bias.data.fill_(0.)

        # ----- Initialize bn5 -----
        self.bn5.weight.data.fill_(1.)
        self.bn5.bias.data.fill_(0.)

        # ----- Initialize GRU -----
        def _concat_init(tensor, init_funcs):
            (length, fan_out) = tensor.shape
            fan_in = length // len(init_funcs)
            for i, init_func in enumerate(init_funcs):
                init_func(tensor[i * fan_in : (i + 1) * fan_in, :])

        def _inner_uniform(tensor):
            fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
            bound = math.sqrt(3.0 / fan_in)
            nn.init.uniform_(tensor, -bound, bound)

        for i in range(self.gru.num_layers):
            # weight_ih_l{i}
            _concat_init(getattr(self.gru, f'weight_ih_l{i}'),
                        [_inner_uniform, _inner_uniform, _inner_uniform])
            getattr(self.gru, f'bias_ih_l{i}').data.fill_(0.)

            # weight_hh_l{i}
            _concat_init(getattr(self.gru, f'weight_hh_l{i}'),
                        [_inner_uniform, _inner_uniform, nn.init.orthogonal_])
            getattr(self.gru, f'bias_hh_l{i}').data.fill_(0.)

        # ----- Initialize fc -----
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            self.fc.bias.data.fill_(0.)

    def forward(self, input):
        """
        Args:
          input: (batch_size, channels_num, time_steps, freq_bins)

        Outputs:
          output: (batch_size, time_steps, classes_num)
        """
        x = self.convBlock1(input, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.convBlock2(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.convBlock3(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.convBlock4(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = x.transpose(1, 2).flatten(2)
        x = F.relu(self.batchnorm5(self.fc5(x).transpose(1, 2)).transpose(1, 2))
        x = F.dropout(x, p=0.5, training=self.training, inplace=True)
        
        (x, _) = self.gru(x)
        x = F.dropout(x, p=0.5, training=self.training, inplace=False)
        output = torch.sigmoid(self.fc(x))
        return output