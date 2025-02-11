import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, TimeDistributed, Flatten

from core.align import Align

class LipReaderModel(object):
    def __init__(self, img_c=3, img_w=100, img_h=50, frames_n=75, align_len=40):
        self.img_c = img_c
        self.img_w = img_w
        self.img_h = img_h
        self.frames_n = frames_n
        self.align_len = align_len

        self.align = Align(align_len)

    @staticmethod
    def CTCLoss(y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
        return loss

    def create_dnn_architecture(self, Conv3D_filters=(128, 256, 75), kernel_size=3, LSTM_filters=(128, 128)):
        input_shape = (self.frames_n, self.img_h, self.img_w, self.img_c)
        model = Sequential()

        # Add three convolution layers with relu activation and max pooling
        model.add(Conv3D(Conv3D_filters[0], kernel_size, input_shape=input_shape, padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPool3D((1, 2, 2)))

        model.add(Conv3D(Conv3D_filters[1], kernel_size, padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPool3D((1, 2, 2)))

        model.add(Conv3D(Conv3D_filters[2], kernel_size, padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPool3D((1, 2, 2)))

        model.add(TimeDistributed(Flatten()))

        # Add two bidirectional LSTM layers
        model.add(Bidirectional(LSTM(LSTM_filters[0], kernel_initializer='Orthogonal', return_sequences=True)))
        model.add(Dropout(.5))

        model.add(Bidirectional(LSTM(LSTM_filters[1], kernel_initializer='Orthogonal', return_sequences=True)))
        model.add(Dropout(.5))

        model.add(Dense(len(self.align.get_vocabulary()) + 1, kernel_initializer='he_normal', activation='softmax'))

        return model