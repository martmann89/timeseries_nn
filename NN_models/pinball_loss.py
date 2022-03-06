# import keras
from keras.layers import Dense, Input, Flatten, LSTM, Conv1D, MaxPool1D
from keras.models import Sequential
from keras.optimizers import Adam
import tensorflow as tf
import config as cfg


def multiple_pinball(alpha, y_true, y_pred):
    # hyperparameters
    alpha_lower = alpha / 2  # for pinball
    alpha_upper = 1 - alpha_lower

    y_true = y_true[:, 0]
    y_l = y_pred[:, 0]
    y_u = y_pred[:, 1]
    y_m = y_pred[:, 2]

    error_l = tf.subtract(y_true, y_l)
    error_m = tf.subtract(y_true, y_m)
    error_u = tf.subtract(y_true, y_u)
    lower_sum = tf.maximum(alpha_lower * error_l, (alpha_lower - 1) * error_l)
    middle_sum = tf.maximum(0.5 * error_m, -0.5 * error_m)
    upper_sum = tf.maximum(alpha_upper * error_u, (alpha_upper - 1) * error_u)
    return tf.reduce_mean((lower_sum + middle_sum + upper_sum) / 3, axis=-1)
    # return tf.reduce_mean((lower_sum + upper_sum) / 2, axis=-1)


def create_pb_model(m_storage):
    alpha = m_storage.get('alpha', cfg.prediction['alpha'])

    model = None
    if m_storage['nn_type'] == 'mlp':
        ### Dense Model
        model = Sequential()
        model.add(Input(shape=(cfg.nn_pred['input_len'], 1)))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(3, activation='linear'))
    elif m_storage['nn_type'] == 'LSTMconv':
        ### LSTMconv Model
        model = Sequential()
        model.add(Input(shape=(cfg.nn_pred['input_len'], 1)))
        model.add(LSTM(32, return_sequences=True))
        model.add(Conv1D(filters=256, activation='relu', kernel_size=3, strides=1, padding='same'))
        model.add(MaxPool1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(3, kernel_initializer=tf.initializers.zeros, activation='linear'))

    model.__setattr__('alpha', alpha)
    model.__setattr__('loss_type', 'pinball')

    def loss_function(y_true, y_pred):
        return multiple_pinball(alpha, y_true, y_pred)

    # opt = keras.optimizers.Adam(lr=0.02, decay=0.01)
    opt = Adam()
    model.compile(
        loss=loss_function,
        optimizer=opt)
    return model
