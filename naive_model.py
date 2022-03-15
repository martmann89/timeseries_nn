import numpy as np
import config as cfg


def naive_prediction(model, data_set, input_len):
    """
    runs naive model as benchmark

    Parameters
    ----------
    model : dict
        dictionary with information of used model, defined in "config_models.py"-file
    data_set : DataFrame
        input data set for network training
    input_len : int
        historic values for min/max calculation

    Returns
    -------
    m_storage: dict
        intervals, inputs, labels added
    """
    _input_len = input_len
    test_len = cfg.data['test_data_size']
    intervals = np.empty((0, 2), float)
    labels = np.empty((0, 1), float)
    inputs = np.empty((0, _input_len, 1), float)
    for i in range(len(data_set) - test_len, len(data_set)):
        train, true = _get_traindata(_input_len, data_set, i)
        intervals = np.append(intervals, np.array([[train.min(), train.max()]]), axis=0)
        labels = np.append(labels, np.array(true).reshape(1, 1), axis=0)
        inputs = np.append(inputs, np.array(train).reshape((1, _input_len, 1)), axis=0)
    model['intervals'] = intervals
    model['labels'] = labels
    model['inputs'] = inputs
    return model


def _get_traindata(length, data, true_idx):
    true_val = data[cfg.label][true_idx]
    train_data = data[cfg.label][true_idx-length:true_idx]
    return train_data, true_val
