import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import config

def segment_events_to_labels(segment_events, n_frames, n_keys=config.num_classes):
    """
    Converts a list of MIDI note events in a segment to frame-wise labels.

    Parameters:
    - segment_events: list of (start_frame, end_frame, pitch, velocity)
    - n_frames: number of frames in the segment (e.g. 1001)
    - n_keys: usually 88 (for piano notes 21 to 108)

    Returns:
    - dict of frame-wise labels for onset, offset, frame, and velocity
    """
    # Initialize label arrays
    reg_onset_output = np.zeros((n_frames, n_keys), dtype=np.float32)
    reg_offset_output = np.zeros((n_frames, n_keys), dtype=np.float32)
    frame_output = np.zeros((n_frames, n_keys), dtype=np.float32)
    velocity_output = np.zeros((n_frames, n_keys), dtype=np.float32)

    for start, end, pitch, velocity in segment_events:
        # MIDI note number → piano key index
        key_index = pitch - 21
        '''
        if not (0 <= key_index < n_keys):
            print(f"[SKIP] Pitch {pitch} → key_index {key_index} is out of range.")
            continue

        print(f"[OK] Pitch {pitch} → key_index {key_index} | Frames: {start} to {end}")        
        '''
        if not (0 <= key_index < n_keys):
            continue  # skip out-of-range notes

        # Clip start and end to the segment range
        start = max(0, min(start, n_frames - 1))
        end = max(0, min(end, n_frames))

        
        if start < end:
            frame_output[start:end, key_index] = 1.0
            velocity_output[start:end, key_index] = velocity / 127.0  # normalize

            # Regressed onset and offset: 1 only at start and end frames
            reg_onset_output[start, key_index] = 1.0
            if end < n_frames:
                reg_offset_output[end, key_index] = 1.0

    return {
        'reg_onset_output': reg_onset_output,
        'reg_offset_output': reg_offset_output,
        'frame_output': frame_output,
        'velocity_output': velocity_output
    }

if __name__ == "__main__":
    # Example usage
    segment_events = [
        (0, 10, 60, 100),  # Note on at frame 0, off at frame 10, pitch 60 (Middle C), velocity 100
        (5, 15, 62, 120)   # Note on at frame 5, off at frame 15, pitch 62 (D#), velocity 120
    ]
    n_frames = config.frames_per_second * config.secment_length
    labels = segment_events_to_labels(segment_events, n_frames)
    #print(labels)
    #print(np.max(labels["frame_output"]))
    #print(np.count_nonzero(labels["frame_output"]))
