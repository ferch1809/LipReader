import os
import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam

from core.align import Align
from core.generators import Generator
from architectures.dnn import LipReaderModel

class ProducePrediction(tf.keras.callbacks.Callback):
    def __init__(self, dataset, align_len) -> None:
        self.dataset = dataset.as_numpy_iterator()
        self.align = Align(align_len)

    def on_epoch_end(self, epoch, logs=None) -> None:
        print(f"\n=== Epoch {epoch} predictions ===")
        data = self.dataset.next()
        yhat = tf.cast(self.model.predict(data[0]), tf.float32)

        # Determine sequence lengths
        batch_size = tf.shape(yhat)[0]
        time_steps = tf.shape(yhat)[1]
        seq_lengths = tf.fill([batch_size], time_steps)
        # Decode the predictions (using greedy decoder)
        decoded = tf.keras.backend.ctc_decode(yhat, seq_lengths, greedy=False)[0][0].numpy()

        # For each sample in the batch, convert numeric labels to characters and print
        for i in range(len(yhat)):
              original = tf.strings.reduce_join(self.align.num_to_char(data[1][i])).numpy().decode('utf-8')
              prediction = tf.strings.reduce_join(self.align.num_to_char(decoded[i])).numpy().decode('utf-8')
              print("Original:   ", original)
              print("Prediction: ", prediction)
              print("~" * 100)

def scheduler(epoch, lr):
            return lr if epoch < 30 else lr * tf.math.exp(-0.1)

def train(dataset_path=None, speakers=None, shuffle_buffer=500, batch_size=2, img_c=3, img_w=100, img_h=50, frames_n=75, align_len=40):
        lip_gen = Generator(dataset_path, batch_size, img_c, img_w, img_h, frames_n, align_len)
        lipreader = LipReaderModel(img_c, img_w, img_h, frames_n, align_len)

        train, test = lip_gen.create_data_pipeline(speakers, batch_size, shuffle_buffer)
        model = lipreader.create_dnn_architecture()
        model.compile(optimizer=Adam(learning_rate=0.0001), loss=lipreader.CTCLoss)

        checkpoint_callback = ModelCheckpoint(os.path.join('models', 'checkpoint.weights.h5'), monitor='loss', save_weights_only=True, save_freq='epoch')
        schedule_callback = LearningRateScheduler(scheduler)
        pred_callback = ProducePrediction(test, align_len)

        model.fit(train, validation_data=test, epochs=100, steps_per_epoch=lip_gen.train_batches, validation_steps=lip_gen.validation_steps, verbose=2, callbacks=[checkpoint_callback, schedule_callback, pred_callback])
        model.summary()