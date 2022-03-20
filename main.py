import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from operator import mod

from NN_models.neural_networks import run_nn
from naive_model import naive_prediction

import utility
from utility import print_mean_stats, plot_intervals, plot_intervals2
import config_models as cfg_mod


def main():
    ###################################### data import #################################
    ### simulated data
    # data = pd.read_pickle('data/pickles/cos_data_1865.pckl')

    ### Real World Data
    data = pd.read_pickle('data/pickles/PV_Daten_returns.pickle')
    data /= np.std(data)

    df = pd.DataFrame(data, columns=['d_glo'])
    df['#day'] = list(map(lambda x: mod(x, 365), range(len(df))))

    #################################### model dictionaries ###########################
    ### distribution models
    # model_garch = cfg_mod.model_garch
    # model_garch_tv = cfg_mod.model_garch_tv_c
    # model_garch_tv = cfg_mod.model_garch_tv_c

    ### load saved distributoin models
    filepath = 'outputs/models/'
    # utility.save_model(model_garch_tv, filepath + model_garch_tv['name'])
    model_garch_tmp = cfg_mod.model_garch
    model_garch_tv_q_tmp = cfg_mod.model_garch_tv_q
    model_garch_tv_c_tmp = cfg_mod.model_garch_tv_c
    model_garch = utility.load_model(filepath+model_garch_tmp['name']+'.pickle')
    model_garch_tv_q = utility.load_model(filepath+model_garch_tv_q_tmp['name']+'.pickle')
    model_garch_tv_c = utility.load_model(filepath+model_garch_tv_c_tmp['name'])

    ### overwrite saved models
    # model_garch['plotting']['color'] = model_garch_tmp['plotting']['color']
    # model_garch_tv_q['plotting']['color'] = model_garch_tv_q_tmp['plotting']['color']
    # model_garch_tv_c['plotting']['color'] = model_garch_tv_c_tmp['plotting']['color']
    #
    # model_garch_tv_q['name'] = model_garch_tv_q_tmp['name']
    # model_garch_tv_c['name'] = model_garch_tv_c_tmp['name']
    #
    # utility.save_model(model_garch, filepath+model_garch['name'])
    # utility.save_model(model_garch_tv_q, filepath+model_garch_tv_q['name'])
    # utility.save_model(model_garch_tv_c, filepath+model_garch_tv_c['name'])

    ### Neural network models
    model_pb_mlp = cfg_mod.model_pb_mlp
    model_pb_lstm = cfg_mod.model_pb_lstm
    model_qd = cfg_mod.model_qd

    ### naive model
    model_naive = cfg_mod.model_naive

    ################################## Run models #####################################
    ### Run GARCH models
    # model_garch = run_garch(df, model_garch)
    # model_garch_tv_q = run_garch_tv(df, model_garch_tv_q)
    # model_garch_tv_c = run_garch_tv(df, model_garch_tv_c)

    ### Run NN Models
    model_pb_mlp = run_nn(df, model_pb_mlp)
    model_pb_lstm = run_nn(df, model_pb_lstm)
    model_qd = run_nn(df, model_qd)

    ### Run naive model
    model_naive = naive_prediction(model_naive, df, 17)

    ################## Plot prediction intervals (multiple forecasts) ################
    ### Plot 1 = 17, Plot 2 = 175, Plot 3 = 210
    start_idx = 17  # between 0 and 365
    end_idx = start_idx+10
    plt.figure(figsize=(12, 7))
    ### choose models
    plot_intervals2(model_garch, start_idx, end_idx,
                    model_garch['inputs'], model_garch['labels'],
                    date_index=data.index[-365:])
    plot_intervals2(model_garch_tv_q, start_idx, end_idx)
    # plot_intervals2(model_garch_tv_c, start_idx, end_idx)
    # plot_intervals2(model_pb_mlp, start_idx, end_idx, model_garch['inputs'], model_garch['labels'])
    plot_intervals2(model_qd, start_idx, end_idx)
    plot_intervals2(model_pb_lstm, start_idx, end_idx)
    plot_intervals2(model_naive, start_idx, end_idx)
    plt.show()

    ################# Plot prediction intervals (single forecasts) ###################
    # for idx_ in range(3, 4):
    #     plt.figure(figsize=(15, 9))
    #     plot_intervals(model_garch, idx_, model_garch['inputs'], model_garch['labels'])
    #     plot_intervals(model_garch_tv, idx_)
    #     plot_intervals(model_pb, idx_, model_pb['inputs'], model_pb['labels'])
    #     plot_intervals(model_pb, idx_)
    #     plot_intervals(model_qd, idx_)
    #     plot_intervals(model_naive, idx_)
    #     plt.show()

    ################################### Plot Parameter ##################################
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

    ################################ Print stats of models ##############################
    print_mean_stats(model_garch)
    print_mean_stats(model_garch_tv_q)
    print_mean_stats(model_garch_tv_c)
    print_mean_stats(model_pb_mlp)
    print_mean_stats(model_pb_lstm)
    print_mean_stats(model_qd)
    print_mean_stats(model_naive)


if __name__ == '__main__':
    main()
