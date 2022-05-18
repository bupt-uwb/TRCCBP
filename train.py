from keras.preprocessing import sequence
from matplotlib import pyplot as plt
from keras.engine.topology import Layer
import tensorflow as tf
# from keras.layers import *
import scipy.io as sio
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report
import numpy as np
from tensorflow.keras.callbacks import ReduceLROnPlateau
import keras.backend as K
import os
from tensorflow.keras.layers import *

class MultiHeadAttention(Layer):
    def __init__(self, output_dim, num_head, **kwargs):
        self.output_dim = output_dim
        self.num_head = num_head
        super(MultiHeadAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='W',
                                 shape=(self.num_head, 3, input_shape[2], self.output_dim),
                                 initializer='uniform',
                                 trainable=True)
        self.Wo = self.add_weight(name='Wo',
                                  shape=(self.num_head * self.output_dim, self.output_dim),
                                  initializer='uniform',
                                  trainable=True)
        self.built = True

    def call(self, x):
        q = K.dot(x, self.W[0, 0])
        k = K.dot(x, self.W[0, 1])
        v = K.dot(x, self.W[0, 2])
        e = K.batch_dot(q, K.permute_dimensions(k, [0, 2, 1]))
        e = e / (self.output_dim ** 0.5)
        e = K.softmax(e)
        outputs = K.batch_dot(e, v)
        for i in range(1, self.W.shape[0]):
            q = K.dot(x, self.W[i, 0])
            k = K.dot(x, self.W[i, 1])
            v = K.dot(x, self.W[i, 2])
            e = K.batch_dot(q, K.permute_dimensions(k, [0, 2, 1]))
            e = e / (self.output_dim ** 0.5)
            e = K.softmax(e)
            o = K.batch_dot(e, v)
            outputs = K.concatenate([outputs, o])
        z = K.dot(outputs, self.Wo)
        return z

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

    def get_config(self):
        config = {"output_dim": self.output_dim,"num_head": self.num_head}
        base_config = super(MultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    # config = tf.compat.v1.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    # session = tf.compat.v1.Session(config=config)

    data = sio.loadmat('wave1_0507uwb.mat')
    X_train = data['waves_train']
    X_test = data['waves_test']

    y_train = data['bps_train']
    y_test = data['bps_test']
    X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))
    X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))


    # model

    model_input = Input(shape = (X_train.shape[1], X_train.shape[2]))

    x = Conv1D(filters=8, kernel_size=3,padding='same')(model_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)

    y = Conv1D(filters=8, kernel_size=5,padding='same')(model_input)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = MaxPooling1D(pool_size=2, strides=2)(y)

    x = Conv1D(filters=16, kernel_size=3,padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(filters=16, kernel_size=3,padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(filters=32, kernel_size=1,padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2, strides=1)(x)
    #
    y = Conv1D(filters=16, kernel_size=5,padding='same')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv1D(filters=16, kernel_size=5,padding='same')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv1D(filters=32, kernel_size=1,padding='same')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = MaxPooling1D(pool_size=2, strides=1)(y)

    x = Concatenate(axis=-1)([x,y])
    x = Dropout(0.4)(x)


    x = GRU(64, return_sequences=True, activation='tanh')(x)
    x = GRU(64, return_sequences=True, activation='tanh')(x)

    x = MultiHeadAttention(64,4)(x)

    x = Dropout(0.2)(x)

    x = Flatten()(x)
    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    out = Dense(2)(x)

    model = Model(inputs=model_input, outputs=out)

    model.summary()

    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['mae'])

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto')

    history = model.fit(X_train,
              y_train,
              epochs=50, batch_size=32,verbose=1,validation_split=0.1, callbacks=[reduce_lr])

    # model.save('0512seluwb.h5')

    out = model.predict(X_test)
    res = np.mean(abs(y_test - out),axis=0)

    print(res)

    # sio.savemat('wave1_0512seluwb_result.mat',{'ids_test':data['ids_test'],
    #                                      'out':out,'y_test':y_test})