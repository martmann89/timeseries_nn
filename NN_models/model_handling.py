# import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping

import NN_models.pinball_loss as pb_loss
import NN_models.qd_loss as qd_loss

import config as cfg


def choose_model_loss(model):
    model_dict = {
        'pinball': pb_loss.create_pb_model,
        'quality_driven': qd_loss.create_qd_model,
    }
    func = model_dict.get(model['loss'], lambda: "Invalid model type")
    return func(model.get('alpha', cfg.prediction['alpha']))


def fit_model(model, window, epochs):
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=cfg.training['patience'],
                                   mode='min')
    history = model.fit(window.train, epochs=epochs,
                        validation_data=window.val,
                        # callbacks=[early_stopping],
                        )
    return history


def build_model(model, window, epochs, path, train=False):
    if train:
        fit_model(model, window, epochs)
        model.save_weights(path)
    else:
        model.load_weights(path).expect_partial()
    # print(model.summary())
    return model


def get_predictions(model, data, scaler):
    # result = model(data).numpy()
    result = model.predict(data)
    return scaler.inverse_transform(result)
    # return np.concatenate((scaler.inverse_transform(result[0]), scaler.inverse_transform(result[1])), axis=1)
