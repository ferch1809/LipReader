import os
import cv2
import random
import imageio
import numpy as np
import tensorflow as tf
from typing import List

from matplotlib import pyplot as plt

from preprocessing.mouth_extraction import extract_mouth_region
from utils.kalman_filter import KalmanFilter

# Vocabulary and lookup layers
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

# Save raw data for testing the pipeline
def save_sample_data(speakers: List[str], batch):
    """
    Given a batch of (frames, alignments), extract a sample and write:
    - A PNG image of a single frame.
    - A GIF animation of one video.
    - A text file with the decoded alignment.
    The files for each speaker are overwritten each time this function runs.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    package_root = os.path.abspath(os.path.join(current_dir, '..'))

    frames, alignments = batch

    # Loop over each speaker
    for speaker in speakers:
        # Create the directory for this speaker
        speaker_dir = os.path.join(package_root, 'data', 'raw', speaker)
        os.makedirs(speaker_dir, exist_ok=True)

        # Randomly choose a sample index from the batch for the speaker
        sample_idx = random.randint(0, len(frames) - 1)
        video = frames[sample_idx]
        alignment = alignments[sample_idx]
        
        # Save a single representative frame
        if hasattr(video, '__len__') and len(video) > 1:
            sample_frame = random.choice(video)
        else:
            sample_frame = video[0]

        # Convert the frame from float values [0, 1] to uint8 [0, 255]
        sample_frame_uint8 = (sample_frame.numpy() * 255).astype(np.uint8)
        frame_path = os.path.join(speaker_dir, "sample_frame.png")
        plt.imsave(frame_path, sample_frame_uint8)
        print(f"Saved sample frame to {frame_path}")

        # Create GIF animation for the video:
        video_frames_uint8 = [(frame.numpy() * 255).astype(np.uint8) for frame in video]
        gif_path = os.path.join(speaker_dir, "sample_video.gif")
        imageio.mimsave(gif_path, video_frames_uint8, fps=10)
        print(f"saved sample video GIF to {gif_path}")

        # Process the alignment:
        alignment_chars = [
            num_to_char(tf.constant(x)).numpy().decode('utf-8')
            for x in alignment.numpy()
        ]
        alignment_str = "".join(alignment_chars)
        alignment_path = os.path.join(speaker_dir, "sample_align.txt")
        with open(alignment_path, 'w') as f:
            f.write(alignment_str)
        print(f"Saved sample alignment to {alignment_path}")

# Function to load the video and preprocess it
def load_video(path:str) -> List[float]:
    cap = cv2.VideoCapture(path)
    kalman_filter = KalmanFilter()

    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        print(f"Warning: No frames found in video: {path}")
        cap.release()
        return tf.constant([])
    
    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            print("Cannot open video. Exiting...")
            break
        # Extract the mouth region of the frame
        mouth_roi = extract_mouth_region(frame, kalman=kalman_filter)
        
        if mouth_roi is None:
            print("Warning: Mouth region extraction failed for one frame. Skipping...")
            continue
        print(f"Shape of frames: {[frame.shape for frame in frames]}")
        frames.append(mouth_roi)
    cap.release()

    if not frames:
        print(f"Warning: No valid frames were extracted from the video: {path}")
        return tf.constant([])
    
    mean = tf.math.reduce_mean(tf.convert_to_tensor(frames, dtype=tf.float32))
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))

    epsilon = 1e-8
    return tf.cast((tf.convert_to_tensor(frames, dtype=tf.float32) - mean), tf.float32) / (std + epsilon)

# Function to load the alignments of index correspondence
def load_alignments(path:str) -> List[str]:
    if not os.path.exists(path):
        print(f"Warning: Alignment file not found: {path}")
        return tf.constant([], dtype=tf.int64)
    
    with open(path, 'r') as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil':
            tokens = [*tokens, ' ', line[2]]
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]

# Function to load the data and correspondingly align the corresponding vocabulary
def load_data(path: tf.Tensor, speaker: tf.Tensor = tf.constant("s1")):
    # Convert the tf.Tensor to a Python string
    path = bytes.decode(path.numpy())
    speaker = bytes.decode(speaker.numpy())

    file_name = os.path.splitext(os.path.basename(path))[0]
    
    # Construct full video path using the speaker available 
    video_path = os.path.join('/mnt/netac_ssd/gridcorpus/videos', speaker, f'{file_name}.mpg')
    # Construct the alignment path relative to the package root, using the speaker available
    alignment_path = os.path.join('/mnt/netac_ssd/gridcorpus/alignments', speaker, 'align', f'{file_name}.align')

    # Load video frames and alignments
    frames = load_video(video_path)
    if frames is None:
        print(f"Warning: Failed to process video: {video_path}")
        return tf.constant([], dtype=tf.float32), tf.constant([], dtype=tf.int64)
    
    try:
        alignments = load_alignments(alignment_path)
    except FileNotFoundError:
        print(f"Warning: Transcript file not found: {alignment_path}")
        alignments = tf.constant([], dtype=tf.int64)

    return frames, alignments

def mappable_function(path: tf.Tensor, speaker: tf.Tensor = tf.constant("s1")) -> List[tf.Tensor]:
    result = tf.py_function(
        lambda p, s: load_data(p, s), 
        [path, speaker], 
        (tf.float32, tf.int64)
    )
    return result