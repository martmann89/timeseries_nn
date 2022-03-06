import numpy as np
import pandas as pd
import pickle
import time
# from scipy.stats import norm

# import arch.data.sp500

import NN_models.model_handling as m_handling
import utility
from preprocessing.WindowGenerator import WindowGenerator
import preprocessing.preprocessing as pp
import config as cfg
import config_models as cfg_mod
from evaluation import print_mean_stats

from NN_models.qd_loss import qd_objective

alpha_ = cfg.prediction['alpha']


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

    if m_storage['conf_alpha'] is not None:
        m_storage = conf_pred_alpha(m_storage)
    else:
        m_storage['model'] = m_handling.build_model(m_handling.choose_model_loss(m_storage),
                                                    m_storage['window'],
                                                    m_storage['epochs'],
                                                    './checkpoints/' + cfg.data['type'] + '/'
                                                    + m_storage['loss'] + '_loss_' + m_storage['nn_type'],
                                                    train=m_storage['train_bool'])

    c_hat = None
    if m_storage['conf_int']:
        c_hat = conf_pred(m_storage)

    intervals_store = np.empty((0, 3), float)
    labels_store = np.empty((0, 1), float)
    inputs_store = np.empty((0, 6, 1), float)
    for batch_data in m_storage['window'].test:
        inputs, labels = batch_data
        intervals = m_handling.get_predictions(m_storage['model'], inputs, m_storage['scaler'])
        if m_storage['conf_int']:
            intervals_store = np.append(intervals_store, conf_adj(intervals, c_hat), axis=0)
        else:
            intervals_store = np.append(intervals_store, intervals, axis=0)
        # labels_store = np.append(labels_store, np.array(labels[:, 0]), axis=0)
        labels_store = np.append(labels_store, np.array(labels), axis=0)
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


def conf_pred_alpha(m_storage):
    m_storage['train_bool'] = True
    for a in m_storage['conf_alpha']:
        m_storage['alpha'] = a
        m_storage['model'] = m_handling.build_model(m_handling.choose_model_loss(m_storage),
                                                    m_storage['window'],
                                                    m_storage['epochs'],
                                                    './checkpoints/' + cfg.data['type'] + '/' + m_storage['name'],
                                                    train=m_storage['train_bool'])
        intervals_store = np.empty((0, 3), float)
        labels_store = np.empty((0, 1), float)
        for batch_data in m_storage['window'].val:
            inputs, labels = batch_data
            intervals = m_handling.get_predictions(m_storage['model'], inputs, m_storage['scaler'])
            # Y = scale_data_back(np.array(labels), m_storage['scaler']).reshape(-1)
            intervals_store = np.append(intervals_store, intervals, axis=0)
            labels_store = np.append(labels_store, np.array(labels), axis=0)
        picp = np.mean(utility.calc_capt(scale_data_back(labels_store, m_storage['scaler']), intervals_store))
        if picp >= 1 - cfg.prediction['alpha']:
            m_storage['alpha'] = a
            break
    return m_storage


def conf_pred(m_storage):
    c_storage = []
    for batch_data in m_storage['window'].val:
        inputs, labels = batch_data
        pred = m_handling.get_predictions(m_storage['model'], inputs, m_storage['scaler'])
        Y = scale_data_back(np.array(labels), m_storage['scaler']).reshape(-1)
        lo = pred[:, 0]
        up = pred[:, 1]
        m = pred[:, 2]
        c_storage.extend(np.maximum((m - Y) / (m - lo), (Y - m) / (up - m)))
    k = int(np.ceil((1 - alpha_) * (len(c_storage) + 1)))
    return np.sort(c_storage)[k]


def conf_adj(intervals, c_hat):
    m = intervals[:, 2]
    lo = m - c_hat * (m - intervals[:, 0])
    up = m + c_hat * (intervals[:, 1] - m)
    return np.array([lo, up, m]).T


def run_ens(data_set, m_storage, ens):
    m_storage['train_bool'] = True

    # start = time.process_time()
    m_storage = run_nn(data_set, m_storage)
    # ende = time.process_time()
    # print('Single NN: {:5.3f}s'.format(ende - start))

    lo = m_storage['intervals'][:, 0]
    up = m_storage['intervals'][:, 1]
    # m = m_storage['intervals'][:, 2]
    for i in range(1, ens):
        print(i)
        m_storage = run_nn(data_set, m_storage)
        lo = np.c_[lo, m_storage['intervals'][:, 0]]
        up = np.c_[up, m_storage['intervals'][:, 1]]
        # m = np.c_[m, m_storage['intervals'][:, 2]]
    # lo_mean = lo.mean(axis=1)
    # lo_var = lo.var(axis=1, ddof=1)
    # lo_int = lo_mean - norm.ppf(1-alpha_/2)*np.sqrt(lo_var)
    # up_mean = up.mean(axis=1)
    # up_var = up.var(axis=1, ddof=1)
    # up_int = up_mean + norm.ppf(1-alpha_/2)*np.sqrt(up_var)
    lo_int = lo.mean(axis=1)
    up_int = up.mean(axis=1)

    m_storage['intervals'] = np.array([lo_int, up_int]).T

    # m_mean = m.mean(axis=1)
    # m_var = m.var(axis=1, ddof=1)
    # m_lo_int = m_mean - norm.ppf(1-alpha_/2)*np.sqrt(m_var)
    # m_up_int = m_mean + norm.ppf(1-alpha_/2)*np.sqrt(m_var)
    # m_storage['simple_approach'] = np.array([m_lo_int, m_up_int]).T
    return m_storage


if __name__ == '__main__':
    ### get and preprocess data
    # data = arch.data.sp500.load()
    # market = pd.DataFrame(data, columns={"Adj Close"})
    # market = market.rename(columns={'Adj Close': 'd_glo'})
    # df = market.diff(1).dropna()
    # file_loc = 'data/pickles/cos_data_1865.pckl'
    file_loc = 'data/pickles/PV_Daten_returns.pickle'
    data = pd.read_pickle(file_loc)
    data /= np.std(data)
    df = pd.DataFrame(data, columns=['d_glo'])
    # df = df.diff(1).dropna()

    # model = cfg_mod.model_pb
    model = cfg_mod.model_qd

    start = time.process_time()
    # model = run_ens(df, model, 100)  # at least 2 repetitions
    model = run_nn(df, model)
    ende = time.process_time()
    print('Ensembled: {:5.3f}s'.format(ende - start))

    # model1 = dict(
    #     intervals=model['conf1_intervals'],
    #     labels=model['labels'],
    #     name='Pinnball Loss Conf1'
    # )

    print_mean_stats(model)
    print('Hello_world')
