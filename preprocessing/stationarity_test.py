import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from statsmodels.tsa.stattools import adfuller, kpss
import matplotlib.pyplot as plt


def main():
    data = pd.read_pickle('data/pickles/PV_Daten_returns.pickle')
    # data = pd.read_pickle('data/pickles/time_varying_data_2230.pckl')
    ### Scaling data (for GARCH)
    scaler = MinMaxScaler(feature_range=(-10, 10))
    data = scaler.fit_transform(data)
    # plt.hist(data)
    # plt.show()
    # X = np.array(data)
    result_adf: tuple = adfuller(data)
    result_kpss: tuple = kpss(data, nlags='auto')

    print('######## ADF Test ########')
    print(results2series(result_adf))
    print('######## KPSS Test ########')
    print(results2series(result_kpss))
    # split = int(len(X) / 2)
    # X1, X2 = X[0:split], X[split:]
    # mean1, mean2 = X1.mean(), X2.mean()
    # var1, var2 = X1.var(), X2.var()
    # print('mean1=%f, mean2=%f' % (mean1, mean2))
    # print('variance1=%f, variance2=%f' % (var1, var2))

    # mean_ls = list()
    # var_ls = list()
    # old = 0
    # n_split = 10
    # for i in range(1, n_split):
    #     b = int(i*len(X)/n_split)
    #     mean_ls.append(X[old:b].mean())
    #     var_ls.append(X[old:b].var())
    #     old = b
    # print('means= ', mean_ls)
    # print('vars= ', var_ls)
    # print('foo')


def results2series(test_res):
    if len(test_res) == 6:
        return pd.Series([test_res[0], test_res[1]] + list(test_res[4].values()),
                         ['test_statistic', 'p-value'] + list(test_res[4].keys()))
    else:
        return pd.Series([test_res[0], test_res[1]] + list(test_res[3].values()),
                         ['test_statistic', 'p-value'] + list(test_res[3].keys()))


if __name__ == '__main__':
    main()
