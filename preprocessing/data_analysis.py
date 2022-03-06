import pandas as pd
import numpy as np

# from sklearn.preprocessing import MinMaxScaler
from scipy.stats import skew

from statsmodels.tsa.stattools import adfuller, kpss
import matplotlib.pyplot as plt
import config as cfg


def main():
    data = pd.read_pickle('data/pickles/PV_Daten_returns.pickle')
    # data = pd.read_pickle('data/pickles/quad_data_1865.pckl')
    # data = pd.read_pickle('data/pickles/quad_data_1865.pckl')
    data = data[cfg.label]
    ### Scaling data
    data /= np.std(data)
    # plt.figure(figsize=(10, 6))
    # plt.plot(data)
    # plt.show()
    skewness_test_monthly(data)
    # eliminate_seasonality(data)
    # stationarity_tests(data)


def skewness_test(data):
    arr = np.array(data)[:-65]
    arr = arr.reshape(30, -1)
    skew_data = skew(arr)
    plt.figure(figsize=(10, 6))
    plt.plot(skew_data)
    # for i in range(5):
    #     plt.plot(skew_data[i*12:(i+1)*12], label='year' + str(i+1))
    plt.xlabel('month')
    plt.ylabel('skewness')
    # plt.legend(loc='best')
    plt.show()


def skewness_test_monthly(data):
    # df_year = data.to_period('Y')
    # data = data[-cfg.data['test_data_size']:]
    # df = data[i*365:((i+1)*365)]
    df = data
    df = df.to_period('M')
    grouped_arr = df.groupby(by=df.index).skew().to_frame(name='month_skew')
    plt.figure(figsize=(10, 6))
    plt.plot(np.array(grouped_arr))
    # for i in range(2013, 2017):
    #     plt.plot(np.array(grouped_arr.iloc[grouped_arr.index.year == i]), label=i)
    plt.xlabel('month')
    plt.ylabel('skewness')
    # plt.legend(loc='lower right')
    plt.show()


def stationarity_tests(data):
    result_adf: tuple = adfuller(data)
    result_kpss: tuple = kpss(data, nlags='auto')

    print('######## ADF Test ########')
    print(results2series(result_adf))
    print('######## KPSS Test ########')
    print(results2series(result_kpss))


def variance_test(data):
    X = np.array(data)
    split = int(len(X) / 2)
    X1, X2 = X[0:split], X[split:]
    mean1, mean2 = X1.mean(), X2.mean()
    var1, var2 = X1.var(), X2.var()
    print('mean1=%f, mean2=%f' % (mean1, mean2))
    print('variance1=%f, variance2=%f' % (var1, var2))

    mean_ls = list()
    var_ls = list()
    old = 0
    n_split = 10
    for i in range(1, n_split):
        b = int(i*len(X)/n_split)
        mean_ls.append(X[old:b].mean())
        var_ls.append(X[old:b].var())
        old = b
    print('means= ', mean_ls)
    print('vars= ', var_ls)
    print('foo')


def results2series(test_res):
    if len(test_res) == 6:
        return pd.Series([test_res[0], test_res[1]] + list(test_res[4].values()),
                         ['test_statistic', 'p-value'] + list(test_res[4].keys()))
    else:
        return pd.Series([test_res[0], test_res[1]] + list(test_res[3].values()),
                         ['test_statistic', 'p-value'] + list(test_res[3].keys()))


def eliminate_seasonality(data):
    X = [i % 365 for i in range(1, len(data) + 1)]
    y = data[cfg.label].values

    degree = 4
    coef = np.polyfit(X, y, degree)
    print('Coefficients: %s' % coef)
    # create curve
    curve = list()
    for i in range(len(X)):
        value = coef[-1]
        for d in range(degree):
            value += X[i] ** (degree - d) * coef[d]
        curve.append(value)
    # plt.plot(y)
    # plt.plot(curve, color='red', linewidth=3)
    # plt.show()

    df = data.diff(1).dropna()
    y_diff = df[cfg.label].values

    # create seasonally adjusted
    diff = list()
    for i in range(len(y)-1):
        # value = y[i] - curve[i]
        value = y_diff[i] / curve[i]
        diff.append(value)
    diff = pd.DataFrame(diff)
    plt.plot(diff)
    plt.show()


if __name__ == '__main__':
    main()
