import matplotlib.pyplot as plt
import pickle

import arch.data.sp500
from arch import arch_model
from SkewStudent import SkewStudent
import numpy as np

import config as cfg


def main():
    horizon_ = 1
    alpha_ = cfg.intervals['alpha']
    length_ = cfg.garch['input_len']
    data = arch.data.sp500.load()
    market = data["Adj Close"]
    # returns = 100 * market.pct_change().dropna()
    returns = market.diff(1).dropna()
    # plt.plot(returns)
    # plt.plot(market)
    # plt.show()
    # ax = returns.plot()
    # xlim = ax.set_xlim(returns.index.min(), returns.index.max())

    def plot_interval(data, idx, interval):
        plt.plot(data[idx-cfg.garch['input_len']:idx])
        ts = data.index[idx]
        plt.plot(ts, data[idx], 'ko')
        plt.plot(ts, interval[0]+data[idx-1], 'gx')
        plt.plot(ts, interval[1]+data[idx-1], 'rx')
        plt.show()

    def get_traindata(length, data, true_idx):
        true = data[true_idx]
        train = data[true_idx-length:true_idx]
        return train, true

    def get_interval(y_true, fit_res, print_results=False):
        eta, lam = fit_res.params['nu'], fit_res.params['lambda']
        skewt_dist = SkewStudent(eta=eta, lam=lam)
        lb, ub = skewt_dist.ppf(alpha_ / 2), skewt_dist.ppf(1 - alpha_ / 2)
        fcast = fit_res.forecast(horizon=horizon_, reindex=False)
        std_dev = np.sqrt(fcast.variance['h.1'][0])
        interval = [std_dev * lb, std_dev * ub]
        if print_results:
            print('True value: ', y_true)
            print(f'Parameter - dof={eta}; lambda={lam}')
            print(f'lower bound quantile = {lb}, upper bound quantile = {ub}')
            print('Standard deviation =', std_dev)
            print(f'Interval: {interval}, width=', interval[1] - interval[0])
        return interval

    # y_true = returns[-1]

    # for i in [len(returns)-1, 150, 100, 75, 50]:
    # for i in [market.index.get_loc('23.03.2018')]:
    intervals = np.empty((0, 2), float)
    labels = np.empty((0, 1), float)
    inputs = np.empty((0, 6, 1), float)
    for i in range(len(returns)-cfg.data['test_data_size']+cfg.prediction['input_len'], len(returns)):
        train, y_true = get_traindata(length_, returns, i)
        # train = returns[(len(returns)-i-1):-1]
        am = arch_model(train, dist="skewt")
        res = am.fit(update_freq=0, disp='off')
        # print('===============================================')
        # # print('Number of observations for fitting: ', i)
        # get_interval(y_true, res, print_results=True)
        interval = get_interval(y_true, res, print_results=False)
        intervals = np.append(intervals, np.array(interval).reshape(1, 2), axis=0)
        labels = np.append(labels, np.array(y_true).reshape(1, 1), axis=0)
        inputs = np.append(inputs, np.array(train[-6:]).reshape(1, 6, 1), axis=0)
        # plot_interval(market, i, interval)
    with open('outputs/intervals/garch_intervals.pickle', 'wb') as f:
        pickle.dump([inputs, labels, intervals], f)
    # print('Range of returns: ', returns.min(), returns.max())


if __name__ == '__main__':
    main()
