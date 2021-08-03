import numpy as np
import pandas as pd
import pickle

import arch.data.sp500

import NN_models.model_handling as m_handling
from preprocessing.WindowGenerator import WindowGenerator
import preprocessing.preprocessing as pp
import config as cfg
import config_models as cfg_mod

from NN_models.qd_loss import qd_objective


def run_nn(data_set, m_storage):
    """
    Returns prediction intervals (at given confidence level) calculated with defined nn-model.
     Moreover the inputs which were used for training and the true value of the predicted interval are returned

    Parameters
    ----------
    data_set : DataFrame
        input data set for network training
    m_storage : dict
        dictionary with information of used model, defined in "config_models.py"-file

    Returns
    -------
    m_storage: dict
        intervals, inputs, labels added
    """
    train, val, test = pp.preprocess(data_set, [cfg.label])
    m_storage['train'], m_storage['val'], m_storage['test'] = train, val, test

    with open('outputs/scaler/output_scaler.pckl', 'rb') as file_scaler:
        m_storage['scaler'] = pickle.load(file_scaler)

    m_storage['window'] = WindowGenerator(input_width=cfg.nn_pred['input_len'],
                                          label_width=cfg.prediction['horizon'],
                                          train_df=train, val_df=val, test_df=test,
                                          shift=cfg.prediction['horizon'],
                                          label_columns=[cfg.label],
                                          )

    m_storage['model'] = m_handling.build_model(m_handling.choose_model_loss(m_storage),
                                                m_storage['window'],
                                                './checkpoints/pv_data/' + m_storage['loss'],
                                                train=m_storage['train_bool'])

    # model = m_handling.choose_model_loss(m_storage)
    # for batch_data in m_storage['window'].train:
    #     inputs, labels = batch_data
    #     test_pred = model.predict(inputs)
    #     loss = qd_objective(labels, test_pred)
    #     print(loss)

    intervals_store = np.empty((0, 2), float)
    labels_store = np.empty((0, 1), float)
    inputs_store = np.empty((0, 6, 1), float)
    for batch_data in m_storage['window'].test:
        inputs, labels = batch_data
        intervals = m_handling.get_predictions(m_storage['model'], inputs, m_storage['scaler'])
        intervals_store = np.append(intervals_store, intervals, axis=0)
        labels_store = np.append(labels_store, np.array(labels[:, 0]), axis=0)
        inputs_store = np.append(inputs_store, np.array(inputs), axis=0)

    # Save in pickles
    # with open('outputs/intervals/' + m_storage['loss'] + '_intervals.pickle', 'wb') as f:
    #     pickle.dump([scale_data_back(inputs_store, m_storage['scaler']),
    #                  scale_data_back(labels_store, m_storage['scaler']),
    #                  intervals_store], f)
    m_storage['intervals'] = intervals_store
    m_storage['inputs'] = scale_data_back(inputs_store, m_storage['scaler'])
    m_storage['labels'] = scale_data_back(labels_store, m_storage['scaler'])
    return m_storage


def scale_data_back(data_arr, scaler):
    if len(data_arr.shape) == 3:
        d1, d2, d3 = data_arr.shape
        data_arr = data_arr.reshape(d1*d2, d3)
        data_arr = scaler.inverse_transform(data_arr)
        return data_arr.reshape(d1, d2, d3)
    else:
        return scaler.inverse_transform(data_arr)


# def plot_interval(input, label, interval, scaler):
#     input = scaler.inverse_transform(input)
#     label = scaler.inverse_transform(label.reshape(-1, 1))
#     plt.plot(input)
#     plt.plot(6, label, 'ko')
#     plt.plot(6, interval[0], 'gx')
#     plt.plot(6, interval[1], 'rx')
#     plt.show()


if __name__ == '__main__':
    # get and preprocess data
    # data = arch.data.sp500.load()
    # market = pd.DataFrame(data, columns={"Adj Close"})
    # market = market.rename(columns={'Adj Close': 'd_glo'})
    # df = market.diff(1).dropna()
    file_loc = 'data/pickles/PV_Daten.pickle'
    df = pd.read_pickle(file_loc)
    df = df.diff(1).dropna()
    model = cfg_mod.model_pb
    model = run_nn(df, model)
    print('Hello_world')
