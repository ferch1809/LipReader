#!/usr/bin/env python3
import tensorflow as tf
tf.keras.backend.clear_session()

# Memory growth for specific GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

from core.callbacks import train

DATASET_PATH = '/mnt/netac_ssd/gridcorpus'
BATCH_SIZE = 1

# Configure dataset, model, and training callbacks
def main():
    speakers = ['s1']
    train(dataset_path=DATASET_PATH, speakers=speakers, batch_size=BATCH_SIZE)

if __name__ == '__main__':
    main()