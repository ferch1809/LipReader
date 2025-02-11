import os
import tensorflow as tf

import imageio
import random
import numpy as np
from typing import List

from core.videos import Video
from core.align import Align

from matplotlib import pyplot as plt

# Class will generate the pipeline for the LipReader model
class Generator(tf.keras.callbacks.Callback):
    def __init__(self, dataset_path, batch_size, img_c, img_w, img_h, frames_n, align_len=40):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.img_c = img_c
        self.img_w = img_w
        self.img_h = img_h
        self.frames_n = frames_n
        self.align_len = align_len

        self.validation_steps = 0
        self.train_batches = 0

        self.face_predictor_path = "~/.dlib/shape_predictor_68_face_landmarks.dat"
        self.align = Align(self.align_len, drop_first_token=True)

    # Function to load the data and correspondingly align the corresponding vocabulary
    def load_data(self, path: str, speaker=None):
        # Convert the tf.Tensor to a Python string
        path = bytes.decode(path.numpy())
        speaker = bytes.decode(speaker.numpy())
        
        file_name = os.path.splitext(os.path.basename(path))[0]
        video = Video(face_predictor_path=self.face_predictor_path)
        
        # Construct full video path using the speaker available 
        video_path = os.path.join(self.dataset_path, 'videos', speaker, f'{file_name}.mpg')
        # Construct the alignment path relative to the package root, using the speaker available
        alignment_path = os.path.join(self.dataset_path, 'alignments', speaker, 'align', f'{file_name}.align')

        # Load video frames and alignments
        frames = video.load_video(video_path)
        if frames is None:
            print(f"Warning: Failed to process video: {video_path}")
            return tf.constant([], dtype=tf.float32), tf.constant([], dtype=tf.int64)
    
        try:
            alignments = self.align.load_alignments(alignment_path)
        except FileNotFoundError:
            print(f"Warning: Transcript file not found: {alignment_path}")
            alignments = tf.zeros([self.align_len], dtype=tf.int64)

        return frames, alignments

    def create_file_list(self, speakers, pattern='*.mpg'):
        file_list = []
        for speaker in speakers:
            speaker_dir = os.path.join(self.dataset_path, 'videos', speaker)
            file_pattern = os.path.join(speaker_dir, pattern)
            for f in tf.io.gfile.glob(file_pattern):
                file_list.append((f, speaker))
        return file_list
    
    def mappable_function(self, path: str, speaker: tf.Tensor) -> List[tf.Tensor]:
        result = tf.py_function(
            lambda p, s: self.load_data(p, s), 
            [path, speaker], 
            (tf.float32, tf.int64)
        )
        return result
      
    def create_data_pipeline(self, speakers: List[str], batch_size=2, shuffle_buffer=500):
        input_shape = [self.frames_n, self.img_h, self.img_w, self.img_c]
        file_list = self.create_file_list(speakers, pattern='*.mpg')
        if not file_list:
            raise ValueError(f"No files found for speakers: {speakers}")
        # Unpack file paths and speakers into two lists
        file_paths, speaker_labels = zip(*file_list)
        # Create a dataset of tuples (file_path, speaker)
        dataset = tf.data.Dataset.from_tensor_slices((list(file_paths), list(speaker_labels)))
        dataset = dataset.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
        # Use lambda to pass file path and speaker to mappable_function
        dataset = dataset.map(
            lambda p, s: self.mappable_function(p, speaker=s),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        # Filter out samples where the alignment tensor is empty
        dataset = dataset.filter(lambda frames, alignments: tf.size(alignments) > 0)
        print(f"Number of samples after filtering: {len(list(dataset.as_numpy_iterator()))}")
        # Adjust padded_shapes as needed
        # - video frames: shape [75, None, None, None] (75 frames per video)
        # - alignments: shape [40] (tokes for each of the alignments)
        dataset = dataset.padded_batch(batch_size, padded_shapes=(input_shape, [self.align_len]))
        
        # Concept of TFDS splits (80% train | 20% validation) for batches
        total_batches = len(file_list) // batch_size
        self.train_batches = int(0.8 * total_batches)
        self.validation_steps = total_batches - self.train_batches

        train_dataset = dataset.take(self.train_batches).repeat()
        val_dataset = dataset.skip(self.train_batches).repeat()

        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
        return train_dataset, val_dataset

    # Save raw data for testing the pipeline
    def save_sample_data(self, speakers: List[str], batch):
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
                self.align.num_to_char(tf.constant(x)).numpy().decode('utf-8')
                for x in alignment.numpy()
            ]
            alignment_str = "".join(alignment_chars)
            alignment_path = os.path.join(speaker_dir, "sample_align.txt")
            with open(alignment_path, 'w') as f:
                f.write(alignment_str)
            print(f"Saved sample alignment to {alignment_path}")