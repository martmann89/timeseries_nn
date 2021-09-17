# import pickle

import arch.data.sp500

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from D_models.garch import run_garch
from D_models.garch_tv import run_garch_tv
from NN_models.neural_networks import run_nn
from evaluation import plot_intervals, print_mean_stats, eval_param_fit
# from preprocessing.data_generation import single_ts_generation

import config as cfg
import config_models as cfg_mod
from sklearn.preprocessing import MinMaxScaler


def main():
    ### data import
    # data = pd.read_pickle('data/pickles/time_varying_data_2230.pckl')
    data = pd.read_pickle('data/pickles/PV_Daten_returns.pickle')

    # data.plot()
    # plt.show()

    ### Scaling data (for GARCH)
    scaler = MinMaxScaler(feature_range=(-10, 10))
    data = scaler.fit_transform(data)

    df = pd.DataFrame(data, columns=['d_glo'])

    ### model dictionaries
    model_garch = cfg_mod.model_garch
    # model_garch_tv = cfg_mod.model_garch_tv
    # model_pb = cfg_mod.model_pb
    # model_qd = cfg_mod.model_qd

    ### Run GARCH models
    model_garch = run_garch(df, model_garch)
    # model_garch_tv = run_garch_tv(df, model_garch_tv)

    ### Run NN Models
    # model_pb = run_nn(df, model_pb)
    # model_qd = run_nn(df, model_qd)

    ### Run naive model
    # model_naive = naive_prediction(df)

    # for idx_ in range(3, 7):
    #     plt.figure(figsize=(15, 9))
    #     plot_intervals(model_garch, idx_, model_garch['inputs'], model_garch['labels'])
    #     plot_intervals(model_garch_tv, idx_)
    #     plot_intervals(model_pb, idx_, model_pb['inputs'], model_pb['labels'])
    #     plot_intervals(model_pb, idx_)
    #     plot_intervals(model_qd, idx_)
    #     plot_intervals(model_naive, idx_)
    #     plt.show()

    model_garch = scale_intervals(model_garch, scaler)
    # model_garch_tv = scale_intervals(model_garch_tv, scaler)
    df = pd.DataFrame(scaler.inverse_transform(df), columns=['d_glo'])
    # print(model_garch['LogLikelihood'].mean())
    print(np.mean(model_garch['LogLikelihood']))
    # print(np.mean(model_garch_tv['LogLikelihood']))

    # columns = ['alpha0', 'alpha1', 'beta1', 'eta', 'lam',
    #            'se_alpha0', 'se_alpha1', 'se_beta1', 'se_eta', 'se_lam',
    #            'r_se_alpha0', 'r_se_alpha1', 'r_se_beta1', 'r_se_eta', 'r_se_lam']
    # param_df = pd.DataFrame(np.c_[model_garch['alpha0'], model_garch['alpha1'], model_garch['beta1'],
    #                               model_garch['etas'], model_garch['lams'],
    #                               model_garch['classic_se'], model_garch['robust_se']], columns=columns)
    # columns = ['alpha0', 'alpha1', 'beta1', 'eta1', 'eta2', 'eta3', 'lam1', 'lam2', 'lam3',
    #            'se_alpha0', 'se_alpha1', 'se_beta1', 'se_eta1', 'se_eta2', 'se_eta3', 'se_lam1', 'se_lam2', 'se_lam3',
    #            'r_se_alpha0', 'r_se_alpha1', 'r_se_beta1',
    #            'r_se_eta1', 'r_se_eta2', 'r_se_eta3', 'r_se_lam1', 'r_se_lam2', 'r_se_lam3']
    # param_df = pd.DataFrame(np.c_[model_garch_tv['params'],
    #                               model_garch_tv['classic_se'],
    #                               model_garch_tv['robust_se']], columns=columns)

    # eval_param_fit(param_df)

    print('Data Boundaries: ', [df[cfg.label].min(), df[cfg.label].max()])
    print_mean_stats(model_garch)
    # print_mean_stats(model_garch_tv)
    # print_mean_stats(model_pb)
    # print_mean_stats(model_qd)
    # print_mean_stats(model_naive)


def scale_intervals(model, scaler):
    model['intervals'] = scaler.inverse_transform(model['intervals'])
    model['labels'] = scaler.inverse_transform(model['labels'])
    return model


def naive_prediction(data_set):
    _input_len = 30
    model = dict(name='Naive')
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


def main_test():
    """
    Build different models and compare PIs on simple test data
    Returns
    -------

    """
    # data import/generating
    data = arch.data.sp500.load()
    market = pd.DataFrame(data, columns={"Adj Close"})
    market = market.rename(columns={'Adj Close': 'd_glo'})
    df = market.diff(1).dropna()
    # file_loc = 'data/pickles/PV_Daten.pickle'
    # df = pd.read_pickle(file_loc)
    # df = df.diff(1).dropna()

    # idx_ = 10

    # model dictionaries
    # model_pb = cfg_mod.model_pb
    # model_qd = cfg_mod.model_qd
    model_garch = cfg_mod.model_garch
    model_garch_tv = cfg_mod.model_garch_tv

    # Run GARCH models
    model_garch = run_garch(df, model_garch)
    model_garch_tv = run_garch_tv(df, model_garch_tv)

    # plt.title('LogLikelihood')
    # plt.plot(model_garch['LogLikelihood'], label='GARCH')
    # plt.plot(model_garch_tv['LogLikelihood'], label='time-varying')
    # plt.legend()
    # plt.show()

    # plt.title('etas')
    # plt.plot(model_garch['etas'], label='GARCH')
    # plt.plot(model_garch_tv['etas'], label='time-varying')
    # plt.legend()
    # plt.show()

    plt.title('Lambdas')
    # plt.plot(model_garch['lams'], label='GARCH')
    plt.plot(model_garch_tv['lams'], label='time-varying')
    plt.legend()
    plt.show()

    # Run NN Models
    # model_pb = run_nn(df, model_pb)
    # model_qd = run_nn(df, model_qd)

    # for idx_ in range(13, 17):
    #     plt.figure(figsize=(15, 9))
    #     plot_intervals(model_garch, idx_, model_garch['inputs'], model_garch['labels'])
    #     plot_intervals(model_garch_tv, idx_)
    #     # plot_intervals(model_pb, idx_, model_pb['inputs'], model_pb['labels'])
    #     # plot_intervals(model_pb, idx_)
    #     # plot_intervals(model_qd, idx_)
    #     plt.show()
    #
    print('Data Boundaries: ', [df[cfg.label].min(), df[cfg.label].max()])
    print_mean_stats(model_garch)
    # print_mean_stats(model_garch_tv)
    # print_mean_stats(model_pb)
    # print_mean_stats(model_qd)


if __name__ == '__main__':
    main()
