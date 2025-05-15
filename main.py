'''
This is the main script, it will contain all the functions that can run to do each part of the project.
this includes:
1. Data Preprocessing
2. training the model
3. testing the model
'''

import os
import sys
from DataLoader.GetWavAndMidiPathFromcsv import GetWavAndMidiPathFromcsv
import torch
import torch.nn as nn
import torch.optim as optim
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from preprocessing.WavSpecAndMidiSegmentor import wav_to_spec
'''
This scrit is the main script for the project, it contains the calls for each step of the pipepline.
These include:
1. Making the hdf5 file from WavSpecAndMidiSegmentor see preprocessing/WavSpecAndMidiSegmentor.py to see what the structure and intent is

'''

from preprocessing.CreateHdf5File import CreateHdf5File


if __name__ == "__main__":

    #make the hdf5 file
    CreateHdf5File("hdf5Files/train_hdf5_file")

    #dataloading



    pass