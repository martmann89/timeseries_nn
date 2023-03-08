import NN_models.pinball_loss as pb_loss
import NN_models.qd_loss as qd_loss


def choose_model_loss(model):
    model_dict = {
        'pinball': pb_loss.create_pb_model,
        'quality_driven': qd_loss.create_qd_model,
    }
    func = model_dict.get(model['loss'], lambda: "Invalid model type")
    return func(model)


def fit_model(model, window, epochs):
    history = model.fit(window.train, epochs=epochs,
                        validation_data=window.val,
                        verbose=1,
                        )
    return history


def build_model(model, window, epochs, path, train=False):
    if train:
        fit_model(model, window, epochs)
        model.save_weights(path)
    else:
        model.load_weights(path).expect_partial()
    return model


def get_predictions(model, data, scaler):
    result = model.predict(data)
    return scaler.inverse_transform(result)
