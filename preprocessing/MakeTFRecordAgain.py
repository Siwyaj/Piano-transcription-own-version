import os
import sys
import csv
import numpy as np
import tensorflow as tf
from librosa import cqt, load
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from functools import partial
import multiprocessing
import pretty_midi
from tqdm import tqdm

def cqt_dual(wav_path):
    y, _ = load(wav_path, sr=config.sample_rate, mono=False)  # Load as stereo explicitly
    if y.ndim == 1:  # mono audio fallback
        y = np.stack([y, y], axis=0)  # duplicate channel to make stereo
    
    inner_cqt = partial(cqt, sr=config.sample_rate, hop_length=config.hop_len,
                        fmin=config.f_min, n_bins=config.n_bins, bins_per_octave=config.bins_per_octave)
    specs = inner_cqt(y[0]), inner_cqt(y[1])
    specs = np.abs(np.stack(specs, axis=-1))  # shape: (freq_bins, time_frames, 2)
    return specs

def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten()))

def labels_to_pianoroll(label_path, num_frames, hop_length, sample_rate, n_pitches=128):
    # label_path: path to your txt label file with onset, offset, pitch
    # num_frames: number of frames in spectrogram
    # hop_length, sample_rate: for time/frame conversion
    # n_pitches: usually MIDI pitch range (0-127)
    
    pianoroll = np.zeros((n_pitches, num_frames), dtype=np.float32)
    with open(label_path, 'r') as f:
        for line in f:
            onset, offset, pitch = line.strip().split('\t')
            onset, offset, pitch = float(onset), float(offset), int(pitch)
            
            # Convert onset and offset time to frame indices
            onset_frame = int(onset * sample_rate / hop_length)
            offset_frame = int(offset * sample_rate / hop_length)
            
            # Clamp to available frames
            onset_frame = max(0, min(onset_frame, num_frames-1))
            offset_frame = max(0, min(offset_frame, num_frames-1))
            
            pianoroll[pitch, onset_frame:offset_frame+1] = 1.0
    return pianoroll


def audio_to_tfrecord(wav_path, output_dir, txt_path, window_size=11):
    try:
        spec = cqt_dual(wav_path).astype(np.float32)
        num_frames = spec.shape[1]
        pianoroll = labels_to_pianoroll(txt_path, num_frames, config.hop_len, config.sample_rate).astype(np.float32)

        offset = window_size // 2

        spec_padded = np.pad(spec, ((0, 0), (offset, offset), (0, 0)), mode='constant')
        pianoroll_padded = np.pad(pianoroll, ((0, 0), (offset, offset)), mode='constant')

        tfrecord_path = os.path.join(output_dir, os.path.basename(wav_path).replace('.wav', '.tfrecords'))
        with tf.io.TFRecordWriter(tfrecord_path) as writer:
            for i in range(offset, offset + num_frames):
                window_spec = spec_padded[:, i - offset:i + offset + 1]
                window_label = pianoroll_padded[:, i - offset:i + offset + 1]

                if not np.isfinite(window_spec).all() or not np.isfinite(window_label).all():
                    print(f"Skipping frame {i} in {wav_path} due to NaN/inf.")
                    continue

                example = tf.train.Example(features=tf.train.Features(feature={
                    'spec': float_feature(window_spec),
                    'label': float_feature(window_label),
                }))
                writer.write(example.SerializeToString())

        return f"✅ Processed {os.path.basename(wav_path)}"
    except Exception as e:
        return f"❌ Error processing {os.path.basename(wav_path)}: {e}"


def process_file(args):
    wav_file, output_dir, midi_file, txt_path = args
    if not os.path.exists(wav_file):
        return f"⚠️ Missing WAV: {wav_file}"
    return audio_to_tfrecord(wav_file, output_dir, txt_path)


def extract_labels_from_midi(midi_file):
	midi_data = pretty_midi.PrettyMIDI(midi_file)
	outputs = []
	for instrument in midi_data.instruments:
		notes = instrument.notes
		for note in notes:
			start = note.start
			end = note.end
			pitch = note.pitch
			velocity = note.velocity
			outputs.append([start, end, pitch])
	outputs.sort(key = lambda elem: elem[0])
	return outputs

def midi_to_label_txt(midi_path, tfrecord_dir):
    datas = extract_labels_from_midi(midi_path)
    basename = os.path.basename(midi_path)
    name, ext = os.path.splitext(basename)
    txt_filename = name + '.txt'
    txt_path = os.path.join(tfrecord_dir, txt_filename)
    with open(txt_path, 'wt', encoding='utf8') as f:
        for data in datas:
            onset, offset, pitch = data
            f.write("{:.6f}\t{:.6f}\t{}\n".format(onset, offset, pitch))
    return txt_path



def MakeTFRecord(tfrecord_dir="TFRecords"):
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", 'dataset', 'maestro-v3.0.0', 'maestro-v3.0.0.csv'))
    print("Making TFRecord...")

    for split in ["train", "test", "validation"]:
        os.makedirs(os.path.join(tfrecord_dir, split), exist_ok=True)

    tasks = []

    with open(csv_path, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        csv_lines = list(csv_reader)
        print(f"CSV header: {csv_lines[0]}")

        for line in csv_lines[1:]:
            split = line[2]
            if split in ["train", "test", "validation"]:
                wav_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'maestro-v3.0.0', line[5]))
                midi_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'maestro-v3.0.0', line[4]))
                output_dir = os.path.join(tfrecord_dir, split)
                txt_path = midi_to_label_txt(midi_file, output_dir)
                tasks.append((wav_file, output_dir, midi_file, txt_path))

    cpu_count = multiprocessing.cpu_count()
    with multiprocessing.Pool(cpu_count) as pool:
        for result in tqdm(pool.imap_unordered(process_file, tasks), total=len(tasks), desc="Processing TFRecords"):
            print(result)




if __name__ == "__main__":
    MakeTFRecord(os.path.abspath(os.path.join(os.path.dirname(__file__), "TFRecords")))
