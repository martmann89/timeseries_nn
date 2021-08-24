import arch.data.sp500

import matplotlib.pyplot as plt
import pandas as pd

from D_models.garch import run_garch
from D_models.garch_tv import run_garch_tv
from NN_models.neural_networks import run_nn
from evaluation import plot_intervals, print_mean_stats

import config as cfg
import config_models as cfg_mod


def main():
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

    # Run GARCH model
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
