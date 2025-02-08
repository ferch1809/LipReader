import os
import tensorflow as tf
from typing import List

from utils.loading_functions import mappable_function

def create_file_list(speakers, pattern='*.mpg'):
    file_list = []
    base_dir = '/mnt/netac_ssd/gridcorpus/videos'
    for speaker in speakers:
        speaker_dir = os.path.join(base_dir, speaker)
        file_pattern = os.path.join(speaker_dir, pattern)
        for f in tf.io.gfile.glob(file_pattern):
            file_list.append((f, speaker))
    return file_list

def data_pipeline_func(speakers: List[str], batch_size=2, shuffle_buffer=500):
    file_list = create_file_list(speakers, pattern='*.mpg')
    if not file_list:
        raise ValueError(f"No files found for speakers: {speakers}")
    # Unpack file paths and speakers into two lists
    file_paths, speaker_labels = zip(*file_list)

    # Create a dataset of tuples (file_path, speaker)
    dataset = tf.data.Dataset.from_tensor_slices((list(file_paths), list(speaker_labels)))
    dataset = dataset.shuffle(shuffle_buffer, reshuffle_each_iteration=False)
    # Use lambda to pass file path and speaker to mappable_function
    dataset = dataset.map(
        lambda p, s: mappable_function(p, speaker=s),
        num_parallel_calls=1
    )
    # Filter out samples where the alignment tensor is empty
    dataset = dataset.filter(lambda frames, alignments: tf.size(alignments) > 0)
    
    # Adjust padded_shapes as needed
    # - video frames: shape [75, None, None, None] (75 frames per video)
    # - alignments: shape [40] (tokes for each of the alignments)
    dataset = dataset.padded_batch(batch_size, padded_shapes=([75, 50, 100, 3], [40]))
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    train = dataset.take(450)
    test = dataset.skip(450)

    return train, test