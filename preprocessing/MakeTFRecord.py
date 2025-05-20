import os
import tensorflow as tf
from tqdm import tqdm
import csv

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):  # if value is tensor
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(audio_bytes, filename):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    feature = {
        'audio_raw': _bytes_feature(audio_bytes),
        'filename': _bytes_feature(filename.encode('utf-8')),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

import librosa
import pretty_midi
import tensorflow as tf
import numpy as np
import os

def encode_midi_to_label(midi_file, sr, segment_start, segment_end):
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

def WavToTFRecord(wav_file, midi_file, writer, segment_duration=5.0, sr=16000):
    audio, _ = librosa.load(wav_file, sr=sr)
    total_duration = librosa.get_duration(y=audio, sr=sr)

    for start in np.arange(0, total_duration - segment_duration, segment_duration):
        end = start + segment_duration

        start_sample = int(start * sr)
        end_sample = int(end * sr)
        audio_segment = audio[start_sample:end_sample]

        label = encode_midi_to_label(midi_file, sr, start, end)

        example = tf.train.Example(features=tf.train.Features(feature={
            'audio': tf.train.Feature(float_list=tf.train.FloatList(value=audio_segment)),
            'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[np.array(label, dtype=np.float32).tobytes()])),
            'segment_start': tf.train.Feature(float_list=tf.train.FloatList(value=[start])),
            'segment_end': tf.train.Feature(float_list=tf.train.FloatList(value=[end]))
        }))
        writer.write(example.SerializeToString())



def TFRecord():
    csv_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", 'dataset', 'maestro-v3.0.0', 'maestro-v3.0.0.csv'))

    # Prepare writers for each split
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'TFRecords'))
    os.makedirs(output_dir, exist_ok=True)
    writers = {
        'train': tf.io.TFRecordWriter(os.path.join(output_dir, 'train.tfrecord')),
        'test': tf.io.TFRecordWriter(os.path.join(output_dir, 'test.tfrecord')),
        'validation': tf.io.TFRecordWriter(os.path.join(output_dir, 'validation.tfrecord'))
    }

    with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        csv_lines = list(csv_reader)

        for line in tqdm(csv_lines[1:100]):
            split = line[2]
            if split in writers:
                wav_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'maestro-v3.0.0', line[5]))
                midi_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'maestro-v3.0.0', line[4]))
                print(f"Processing {split}: {wav_file}")
                output_dir = os.path.join(output_dir, f'{split}.tfrecord')
                WavToTFRecord(wav_file, midi_file, writers[split])


    # Close all writers
    for w in writers.values():
        w.close()

if __name__ == "__main__":
    # Create TFRecord files from WAV files
    TFRecord()
    pass