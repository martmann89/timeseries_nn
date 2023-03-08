import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from operator import mod

from NN_models.neural_networks import run_nn
from naive_model import naive_prediction

import utility
from utility import print_mean_stats, plot_intervals, plot_intervals2
import config_models as cfg_mod
from utility import ROOT_DIR

def main():
    ###################################### data import #################################
    ### Real World Data
    data = pd.read_csv(ROOT_DIR + "/data/TTF_FM_new.csv",sep=";")
    df = pd.DataFrame(data, columns=["Price"])
    print(np.std(data))
    #print(data.head(n=40))
    #data /= np.std(data)
    #print(data.head(n=40))
    # df = pd.DataFrame(data, columns=['d_glo'])
    # df['#day'] = list(map(lambda x: mod(x, 365), range(len(df))))

    #################################### model dictionaries ###########################
    ### distribution models
    # model_garch = cfg_mod.model_garch
    # model_garch_tv = cfg_mod.model_garch_tv_c
    # model_garch_tv = cfg_mod.model_garch_tv_c
    doTrain = True

    ### Neural network models
    model_pb_mlp = cfg_mod.model_pb_mlp
    model_pb_lstm = cfg_mod.model_pb_lstm
    model_qd = cfg_mod.model_qd
    if doTrain:
        model_pb_mlp["train_bool"] = True
        model_pb_lstm["train_bool"] = True
        model_qd["train_bool"] = True
    else:
        model_pb_mlp["train_bool"] = False
        model_pb_lstm["train_bool"] = False
        model_qd["train_bool"] = False

    ### naive model
    model_naive = cfg_mod.model_naive

    
    ### Run NN Models
   # model_pb_mlp = run_nn(df, model_pb_mlp)
    model_pb_lstm = run_nn(df, model_pb_lstm)
    model_qd = run_nn(df, model_qd)

    ### Run naive model
    model_naive = naive_prediction(model_naive, df, 17)

    ################## Plot prediction intervals (multiple forecasts) ################
    ### Plot 1 = 17, Plot 2 = 175, Plot 3 = 210
    start_idx = 200  # between 0 and 365
    end_idx = 365
    plt.figure(figsize=(12, 3))
    ### choose models
    # plot_intervals2(model_garch, start_idx, end_idx,
    #                 model_garch['inputs'], model_garch['labels'],
    #                 date_index=data.index[-365:])
    # plot_intervals2(model_garch_tv_q, start_idx, end_idx)
    # plot_intervals2(model_garch_tv_c, start_idx, end_idx)
    # plot_intervals2(model_pb_mlp, start_idx, end_idx, model_garch['inputs'], model_garch['labels'])
    plot_intervals2(model_qd, start_idx, end_idx)
    plot_intervals2(model_pb_lstm, start_idx, end_idx)
    plot_intervals2(model_naive, start_idx, end_idx)
    plt.plot(range(13,178), df[(df.size-165):df.size])
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
    # print_mean_stats(model_garch)
    # print_mean_stats(model_garch_tv_q)
    # print_mean_stats(model_garch_tv_c)
   # print_mean_stats(model_pb_mlp)
    print_mean_stats(model_pb_lstm)
    print_mean_stats(model_qd)
    print_mean_stats(model_naive)


if __name__ == '__main__':
    main()