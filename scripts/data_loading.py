#!/usr/bin/env python3
import os
import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
from data.data_pipeline import data_pipeline_func

from utils.loading_functions import save_sample_data
from architectures.dnn import create_dnn_architecture

# Configure the GPU for loading the program
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

def main():
    print("GPUs:", tf.config.list_physical_devices('GPU'))

    speakers = ['s1', 's2']

    dataset, _, _ = data_pipeline_func(speakers)

    # Get a batch from the dataset
    batch = next(iter(dataset))
    print("Frames shape:", batch[0].shape)
    print("Alignments shape:", batch[1].shape)

    save_sample_data(speakers=speakers, batch=batch)

    model = create_dnn_architecture()
    model.summary()

if __name__ == '__main__':
    main()