# import keras
from keras.layers import Dense, Input, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
import tensorflow as tf
import config as cfg

# hyperparameters
alpha_ = cfg.prediction['alpha']  # capturing (1-alpha)% of samples, for qd
alpha_lower_ = alpha_ / 2  # for pinball
alpha_upper_ = 1 - alpha_lower_


def pinball_loss_upper(y_true, y_pred):
    error = tf.subtract(y_true, y_pred)
    return tf.reduce_mean(tf.maximum(alpha_upper_ * error, (alpha_upper_ - 1) * error), axis=-1)


def pinball_loss_lower(y_true, y_pred):
    error = tf.subtract(y_true, y_pred)
    return tf.reduce_mean(tf.maximum(alpha_lower_ * error, (alpha_lower_ - 1) * error), axis=-1)


def multiple_pinball(y_true, y_pred):
    y_true = y_true[:, 0]
    y_l = y_pred[:, 0]
    y_u = y_pred[:, 1]
    error_l = tf.subtract(y_true, y_l)
    error_u = tf.subtract(y_true, y_u)
    lower_sum = tf.maximum(alpha_lower_ * error_l, (alpha_lower_ - 1) * error_l)
    upper_sum = tf.maximum(alpha_upper_ * error_u, (alpha_upper_ - 1) * error_u)
    return tf.reduce_mean((lower_sum + upper_sum) / 2, axis=-1)


def create_pb_model():
    model = Sequential()
    model.add(Input(shape=(cfg.nn_pred['input_len'], 1)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(2, activation='linear'))

    # opt = keras.optimizers.Adam(lr=0.02, decay=0.01)
    opt = Adam()
    model.compile(
        loss=multiple_pinball,
        optimizer=opt)
    return model

# def create_pb_model():
#     input_layer = keras.Input(shape=(cfg.prediction['input_len'],), name="input")
#     dense_layer = Dense(100, activation='relu')(input_layer)
#     dense_layer = Dense(100, activation='relu')(dense_layer)
#     dense_layer = Dense(100, activation='relu')(dense_layer)
#     lower_pred = Dense(1, activation='linear', name='lower')(dense_layer)
#     upper_pred = Dense(1, activation='linear', name='upper')(dense_layer)
#     model = keras.Model(
#         inputs=[input_layer],
#         outputs=[lower_pred, upper_pred],
#     )
#
#     model.compile(
#         loss={
#             'upper': pinball_loss_upper,
#             'lower': pinball_loss_lower,
#         },
#         optimizer='adam')
#     return model
