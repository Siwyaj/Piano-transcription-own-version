sample_rate = 44100
hop_len = 512
batch_size = 32
secment_length = 10
window_size = 2048
frames_per_second = 100
num_classes = 88
mel_bins = 356
n_bins = 356
win_len = 9
f_min = 27.5
bins_per_octave = 48
max_len = 101
onset_label_len = 7
n_pitches = 88       # MIDI piano range
frame_hop = hop_len / sample_rate  # seconds per frame
segment_duration = 5.0


#from different project
# train params and tfrecord files config
# you need to configurate the tfrd_path as the root path to
# save onset model and pitch model training and evaluation tfrecords. 
#train_params = EasyDict({'train_onset_examples': 15177059, 'train_pitch_examples': 369800, 'initial_lr': 0.0005,
#                         'save_checkpoints_steps': 2000, 'decay_steps': 1000, 'decay_rate': 0.98,
#                         'tfrd_path': 'need to complete!', 'parallel_num': 8})
