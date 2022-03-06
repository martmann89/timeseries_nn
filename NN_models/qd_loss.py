from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, LSTM, Conv1D, MaxPool1D
import keras
import tensorflow as tf
# import numpy as np

import config as cfg

lambda_ = 0.08  # 0.08  # lambda in loss fn  (0.08 for MLP)
# alpha_ = cfg.prediction['alpha']  # capturing (1-alpha)% of samples, for qd
# alpha_lower_ = alpha_ / 2  # for pinball
# alpha_upper_ = 1 - alpha_lower_
soften_ = 100.
n_ = cfg.data['batch_size']  # batch size
# n_epochs_ = 1000


# define loss fn
def qd_objective(m_storage, y_true, y_pred):
    alpha = m_storage.get('alpha', cfg.prediction['alpha'])

    """Loss_QD-soft, from algorithm 1"""
    # print(y_pred.shape)
    y_true = y_true[:, 0]
    # print(y_true.shape)

    y_l = y_pred[:, 0]
    y_u = y_pred[:, 1]
    # y_m = y_pred[:, 2]

    K_HU = tf.maximum(0., tf.sign(y_u - y_true))
    # print(tf.sign(y_u - y_true).shape)
    K_HL = tf.maximum(0., tf.sign(y_true - y_l))
    K_H = tf.multiply(K_HU, K_HL)
    # print(K_H.shape)

    K_SU = tf.sigmoid(soften_ * (y_u - y_true))
    K_SL = tf.sigmoid(soften_ * (y_true - y_l))
    K_S = tf.multiply(K_SU, K_SL)

    MPIW_c = tf.divide(tf.reduce_sum(tf.multiply((y_u - y_l), K_H)), (tf.reduce_sum(K_H)+0.001))
    # PICP_H = tf.reduce_mean(K_H)
    PICP_S = tf.reduce_mean(K_S)
    # print(PICP_S.shape)

    penalty = 10000*tf.maximum(0., y_l-y_u)
    Loss_S = MPIW_c + lambda_ * n_ / (alpha * (1 - alpha)) * tf.square(tf.maximum(0., (1 - alpha) - PICP_S)) + penalty
    # print('Loss=', Loss_S.shape)
    return Loss_S


def create_qd_model(alpha):
    ### Dense Model
    model = Sequential()
    model.add(Input(shape=(cfg.nn_pred['input_len'], 1)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.2)))
    model.add(Dense(3, activation='linear',
                    kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.3),
                    bias_initializer=keras.initializers.Constant(
                        value=[0, 1, 0])))  # important to init biases to start!

    ### LSTMconv Model
    # model = Sequential()
    # model.add(Input(shape=(cfg.nn_pred['input_len'], 1)))
    # model.add(LSTM(32, return_sequences=True))
    # model.add(Conv1D(filters=256, activation='relu', kernel_size=3, strides=1, padding='same'))
    # model.add(MaxPool1D(pool_size=2))
    # model.add(Flatten())
    # model.add(Dense(100, activation='relu', kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.2)))
    # model.add(Dense(3, activation='linear',
    #                 kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.3),
    #                 bias_initializer=keras.initializers.Constant(
    #                     value=[0, 1, 0])))

    def loss_function(y_true, y_pred):
        return qd_objective(alpha, y_true, y_pred)

    # compile
    opt = keras.optimizers.Adam(lr=0.02, decay=0.01)
    model.compile(
        loss=loss_function,
        optimizer=opt)
    return model
