# from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

import models.pinball_loss as pb_loss
import models.qd_loss as qd_loss

import config as cfg


def choose_model_loss(model):
    model_dict = {
        'pinball': pb_loss.create_pb_model,
        'quality_driven': qd_loss.create_qd_model,
    }
    func = model_dict.get(model['loss'], lambda: "Invalid model type")
    return func()


def fit_model(model, window):
    # early_stopping = EarlyStopping(monitor='val_loss',
    #                  patience=cfg.training['patience'],
    #                  mode='min')

    history = model.fit(window.train, epochs=cfg.training['max_epochs'],
                        validation_data=window.val,
                        # callbacks=[early_stopping],
                        )
    return history


def build_model(model, window, path, train=False):
    if train:
        fit_model(model, window)
        model.save_weights(path)
    else:
        model.load_weights(path).expect_partial()
    print(model.summary())
    return model


def get_predictions(model, data, scaler):
    # result = model(data).numpy()
    result = model.predict(data)
    return scaler.inverse_transform(result)
    # return np.concatenate((scaler.inverse_transform(result[0]), scaler.inverse_transform(result[1])), axis=1)
