# import pickle

import arch.data.sp500

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from operator import mod

from D_models.garch import run_garch
from D_models.garch_tv import run_garch_tv
from D_models.my_garch import run_my_garch
from NN_models.neural_networks import run_nn
from evaluation import plot_intervals2, print_mean_stats, eval_param_fit
# from preprocessing.data_generation import single_ts_generation

import utility
import config as cfg
import config_models as cfg_mod
# from sklearn.preprocessing import MinMaxScaler


def main():
    ### data import
    # data = pd.read_pickle('data/pickles/cos_data_1865.pckl')
    data = pd.read_pickle('data/pickles/PV_Daten_returns.pickle')
    # data = pd.read_pickle('data/pickles/solar_irradiation_returns.pickle')
    data /= np.std(data)

    # data.plot(legend=None)
    # plt.show()

    df = pd.DataFrame(data, columns=['d_glo'])
    df['#day'] = list(map(lambda x: mod(x, 365), range(len(df))))

    ### model dictionaries
    # model_garch = cfg_mod.model_garch
    # model_garch_tv = cfg_mod.model_garch_tv_c
    # model_garch_tv = cfg_mod.model_garch_tv_c
    # model_my_garch = cfg_mod.model_my_garch
    model_pb_mlp = cfg_mod.model_pb_mlp
    model_pb_lstm = cfg_mod.model_pb_lstm
    model_qd = cfg_mod.model_qd
    model_naive = cfg_mod.model_naive

    ### Run GARCH models
    # model_garch = run_garch(df, model_garch)
    # model_garch_tv = run_garch_tv(df, model_garch_tv)
    # model_my_garch = run_my_garch(df, model_my_garch)

    ### Run NN Models
    # model_pb_mlp = run_nn(df, model_pb_mlp)
    model_pb_lstm = run_nn(df, model_pb_lstm)
    model_qd = run_nn(df, model_qd)

    ### Run naive model
    model_naive = naive_prediction(model_naive, df, 17)

    model_garch_tmp = cfg_mod.model_garch
    model_garch_tv_q_tmp = cfg_mod.model_garch_tv_q
    # model_garch_tv_c_tmp = cfg_mod.model_garch_tv_c

    filepath = 'outputs/models/'
    # utility.save_model(model_garch_tv, filepath+model_garch_tv['name'])
    model_garch = utility.load_model(filepath+model_garch_tmp['name']+'.pickle')
    model_garch_tv_q = utility.load_model(filepath+model_garch_tv_q_tmp['name']+'.pickle')
    # model_garch_tv_c = utility.load_model(filepath+model_garch_tv_c_tmp['name'])

    # model_garch['plotting']['color'] = model_garch_tmp['plotting']['color']
    # model_garch_tv_q['plotting']['color'] = model_garch_tv_q_tmp['plotting']['color']
    # model_garch_tv_c['plotting']['color'] = model_garch_tv_c_tmp['plotting']['color']

    # model_garch_tv_q['name'] = model_garch_tv_q_tmp['name']
    # model_garch_tv_c['name'] = model_garch_tv_c_tmp['name']
    #
    # utility.save_model(model_garch, filepath+model_garch['name'])
    # utility.save_model(model_garch_tv_q, filepath+model_garch_tv_q['name'])
    # utility.save_model(model_garch_tv_c, filepath+model_garch_tv_c['name'])

    # data[-365:].plot()
    # plt.show()

    ### Plot 1 = 17, Plot 2 = 175, Plot 3 = 210
    start_idx = 17  # between 0 and 365
    end_idx = start_idx+10

    # data.index[-365:]
    plt.figure(figsize=(12, 7))
    plot_intervals2(model_garch, start_idx, end_idx,
                    model_garch['inputs'], model_garch['labels'],
                    date_index=data.index[-365:])
    plot_intervals2(model_garch_tv_q, start_idx, end_idx)
    # plot_intervals2(model_garch_tv_c, start_idx, end_idx)
    # plot_intervals2(model_naive, start_idx, end_idx)
    # plt.show()

    # plot_intervals2(model_pb_mlp, start_idx, end_idx, model_garch['inputs'], model_garch['labels'])
    plot_intervals2(model_qd, start_idx, end_idx)
    plot_intervals2(model_pb_lstm, start_idx, end_idx)
    plot_intervals2(model_naive, start_idx, end_idx)
    plt.show()

    # for idx_ in range(3, 4):
    #     plt.figure(figsize=(15, 9))
    #     plot_intervals(model_garch, idx_, model_garch['inputs'], model_garch['labels'])
        # plot_intervals(model_garch_tv, idx_)
        # plot_intervals(model_pb, idx_, model_pb['inputs'], model_pb['labels'])
        # plot_intervals(model_pb, idx_)
        # plot_intervals(model_qd, idx_)
        # plot_intervals(model_naive, idx_)
        # plt.show()

    # plt.title('etas')
    # plt.plot(model_garch['etas'], label='GARCH')
    # plt.plot(model_garch_tv['etas'], label='time-varying')
    # plt.legend()
    # plt.show()
    #
    # plt.title('Lambdas')
    # # plt.plot(model_garch['lams'], label='GARCH')
    # plt.plot(model_garch_tv['lams'], label='time-varying')
    # plt.legend()
    # plt.show()

    # model_garch = scale_intervals(model_garch, scaler)
    # model_garch_tv = scale_intervals(model_garch_tv, scaler)
    # df = pd.DataFrame(scaler.inverse_transform(df), columns=['d_glo'])
    # print(model_garch['LogLikelihood'].mean())
    # print(np.mean(model_garch['LogLikelihood']))
    # print(np.mean(model_garch_tv['LogLikelihood']))

    ### GARCH
    # columns = ['alpha0', 'alpha1', 'beta1', 'eta', 'lam',
    #            'se_alpha0', 'se_alpha1', 'se_beta1', 'se_eta', 'se_lam',
    #            'r_se_alpha0', 'r_se_alpha1', 'r_se_beta1', 'r_se_eta', 'r_se_lam',
    #            'llh']
    # param_df = pd.DataFrame(np.c_[model_garch['alpha0'], model_garch['alpha1'], model_garch['beta1'],
    #                               model_garch['etas'], model_garch['lams'],
    #                               model_garch['classic_se'], model_garch['robust_se'],
    #                               model_garch['LogLikelihood']], columns=columns)
    ### TV GARCH
    # columns = ['alpha0', 'alpha1', 'beta1', 'eta1', 'eta2', 'eta3', 'lam1', 'lam2', 'lam3',
    #            'se_alpha0', 'se_alpha1', 'se_beta1', 'se_eta1', 'se_eta2', 'se_eta3', 'se_lam1', 'se_lam2', 'se_lam3',
    #            'r_se_alpha0', 'r_se_alpha1', 'r_se_beta1',
    #            'r_se_eta1', 'r_se_eta2', 'r_se_eta3', 'r_se_lam1', 'r_se_lam2', 'r_se_lam3',
    #            'llh', 'eta', 'lam']
    # param_df = pd.DataFrame(np.c_[model_garch_tv['params'],
    #                               model_garch_tv['classic_se'],
    #                               model_garch_tv['robust_se'],
    #                               model_garch_tv['LogLikelihood'],
    #                               model_garch_tv['etas'],
    #                               model_garch_tv['lams']], columns=columns)
    #
    # filename = 'tv_param_cos_365_test_2'
    # with pd.ExcelWriter('outputs/' + cfg.data['type'] + '/' + filename + '.xlsx') as writer:
    #     param_df.to_excel(writer)
    # eval_param_fit(param_df, filename)

    # print('Data Boundaries: ', [df[cfg.label].min(), df[cfg.label].max()])
    # print_mean_stats(model_garch)
    # print_mean_stats(model_garch_tv)
    # print_mean_stats(model_my_garch)
    # print_mean_stats(model_pb)
    # print_mean_stats(model_qd)
    # print_mean_stats(model_naive)


def scale_intervals(model, scaler):
    model['intervals'] = scaler.inverse_transform(model['intervals'])
    model['labels'] = scaler.inverse_transform(model['labels'])
    return model


def naive_prediction(model, data_set, input_len):
    # _input_len = 17
    _input_len = input_len
    # model = dict(name='Naive')
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


if __name__ == '__main__':
    main()
