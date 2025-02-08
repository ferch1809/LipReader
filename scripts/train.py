#!/usr/bin/env python3
import os
import tensorflow as tf

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


from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam

from architectures.dnn import create_dnn_architecture
from data.data_pipeline import data_pipeline_func

from architectures.dnn import CTCLoss
from architectures.dnn import scheduler
from architectures.dnn import ProduceExample

# Configure dataset, model, and training callbacks
def main():
    speakers = ['s1']
    train, test = data_pipeline_func(speakers=speakers, batch_size=1)

    model = create_dnn_architecture()
    model.compile(optimizer=Adam(learning_rate=0.0001), loss=CTCLoss)

    checkpoint_callback = ModelCheckpoint(os.path.join('models', 'checkpoint.weights.h5'), monitor='loss', save_weights_only=True)
    schedule_callback = LearningRateScheduler(scheduler)
    example_callback = ProduceExample(test)

    model.fit(train, validation_data=test, epochs=40, callbacks=[checkpoint_callback, schedule_callback, example_callback])

if __name__ == '__main__':
    main()