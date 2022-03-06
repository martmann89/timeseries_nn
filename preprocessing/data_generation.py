import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import mean_squared_error as mse
from operator import mod

# from NN_models.qd_loss import qd_objective
from D_models.SkewStudent import SkewStudent as SkSt

from D_models.garch_tv import run_single_garch_tv
from D_models.garch import run_single_garch
from utility import exp_trafo, law_of_motion, save_df

import config as cfg

import time


def main():
    np.random.seed(seed=cfg.seed)
    filepath = 'outputs/' + cfg.data['type'] + '/param_' + cfg.data_gen['lom'] + '_2000.pckl'
    # filepath = 'data/pickles/cos_data_1865.pckl'
    # n_mc = cfg.monte_carlo
    # n_mc = 2000
    # data = eval_garch_tv(n_mc)
    # data = eval_garch(n_mc)

    # print('Hallo')
    # start = time.process_time()
    # data, eta_m, lam_m = single_ts_generation(0, 1, 1865)
    # ende = time.process_time()
    # print('Datengenerierung: {:5.3f}s'.format(ende - start))
    # print('eta_mean: ', eta_m)
    # print('lam_mean: ', lam_m)
    # plt.plot(data[cfg.label])
    # plt.show()
    # save_df(data, filepath)

    # start = time.process_time()
    # test = run_single_garch_tv(data)
    # ende = time.process_time()
    # print('Prozesszeit: {:5.3f}s'.format(ende - start))

    # with open(filepath, 'rb') as file_scaler:
    #     df = pickle.load(file_scaler)
    # print(df)

    # plt.show()
    # print(df[['lam1', 'lam2', 'lam3']].mean())
    # print(mse(np.array([np.full(n_mc, b1), np.full(n_mc, b2), np.full(n_mc, b3)]).transpose(),
    #           df[['lam1', 'lam2', 'lam3']],
    #           multioutput='raw_values'))
    # x0 = 0
    # sigma0 = 1
    # n = 1500
    # x, _, _ = single_ts_generation(x0, sigma0, n)
    # x[cfg.label].plot()
    # plt.show()
    # params, llh = run_single_garch_tv(x)


def eval_garch(n_mc):
    par_cols = ['alpha0', 'alpha1', 'beta1', 'eta', 'lam', 'eta_mean', 'lam_mean']
    llh_col = ['llh']
    se_cols = ['se_alpha0', 'se_alpha1', 'se_beta1', 'se_eta', 'se_lam']
    r_se_cols = ['r_se_alpha0', 'r_se_alpha1', 'r_se_beta1', 'r_se_eta', 'r_se_lam']
    cols = par_cols + llh_col + se_cols + r_se_cols
    x0 = 0
    sigma0 = 1
    n = cfg.data_gen['length']
    df = pd.DataFrame(columns=cols)
    for i in range(n_mc):
        print(i)
        x, eta_mean, lam_mean = single_ts_generation(x0, sigma0, n)
        param, llh, std_err, robust_std_err = run_single_garch(x)
        full_dict = {**param,
                     **dict(eta_mean=eta_mean, lam_mean=lam_mean),
                     **dict(zip(llh_col, [llh])),
                     **dict(zip(se_cols, std_err)),
                     **dict(zip(r_se_cols, robust_std_err)),
                     }
        df = df.append(full_dict, ignore_index=True)
    return df


def eval_garch_tv(n_mc):
    par_cols = ['alpha0', 'alpha1', 'beta1', 'eta1', 'eta2', 'eta3', 'lam1', 'lam2', 'lam3']
    llh_col = ['llh']
    se_cols = ['se_alpha0', 'se_alpha1', 'se_beta1', 'se_eta1', 'se_eta2', 'se_eta3', 'se_lam1', 'se_lam2', 'se_lam3']
    r_se_cols = ['r_se_alpha0', 'r_se_alpha1', 'r_se_beta1', 'r_se_eta1', 'r_se_eta2', 'r_se_eta3',
                 'r_se_lam1', 'r_se_lam2', 'r_se_lam3']
    cols = par_cols + llh_col + se_cols + r_se_cols
    x0 = 0
    sigma0 = 1
    n = cfg.data_gen['length']
    df = pd.DataFrame(columns=cols)
    for i in range(n_mc):
        print(i)
        x, eta_mean, lam_mean = single_ts_generation(x0, sigma0, n)
        param, llh, std_err, robust_std_err = run_single_garch_tv(x)
        full_dict = {**param,
                     # **dict(eta_mean=eta_mean, lam_mean=lam_mean),
                     **dict(zip(llh_col, [llh])),
                     **dict(zip(se_cols, std_err)),
                     **dict(zip(r_se_cols, robust_std_err))
                     }
        df = df.append(full_dict, ignore_index=True)
    return df


def single_ts_generation(x0, sigma0, n):
    df = pd.DataFrame(list(map(lambda y: mod(y, 365), range(n))), columns=['#day'])
    x = np.zeros(n)
    x[0] = x0
    df[cfg.label] = x

    sigma2 = sigma0

    # Parameter definition
    # GARCH(1,1)
    alpha0 = cfg.data_gen['alpha0']
    alpha1 = cfg.data_gen['alpha1']
    beta1 = cfg.data_gen['beta1']

    # time-varying dof (eta)
    eta1 = cfg.data_gen['eta1']
    eta2 = cfg.data_gen['eta2']
    eta3 = cfg.data_gen['eta3']

    # time-varying skewness (lambda)
    lam1 = cfg.data_gen['lam1']
    lam2 = cfg.data_gen['lam2']
    lam3 = cfg.data_gen['lam3']
    eta_mean = 0
    lam_mean = 0
    if cfg.data_gen['lom'] == 'quad':
        for i in range(1, n):
            # j = i-1 if cfg.data_gen['lom'] == 'quad' else i
            sigma2 = alpha0 + alpha1 * x[i - 1]**2 + beta1 * sigma2
            # eta_hat = eta1 + eta2 * x[i - 1] + eta3 * x[i - 1] ** 2
            # eta_hat = eta1 + eta2 * np.cos(((df['#day'][i] - eta3) / 365) * 2 * np.pi)
            eta_hat = law_of_motion(df, eta1, eta2, eta3)[i-1]
            eta = exp_trafo(eta_hat, 2.01, 30)
            eta_mean += eta
            # lam_hat = lam1 + lam2 * x[i - 1] + lam3 * x[i - 1] ** 2
            # lam_hat = lam1 + lam2 * np.cos(((df['#day'][i] - lam3) / 365) * 2 * np.pi)
            lam_hat = law_of_motion(df, lam1, lam2, lam3)[i-1]
            # lam_hat = law_of_motion(df, lam1, lam2, lam3)[i]
            lam = exp_trafo(lam_hat, -0.99, 0.99)
            lam_mean += lam
            # print((eta, lam))
            z = SkSt(eta, lam).rvs(1)
            x[i] = np.sqrt(sigma2) * z
            df[cfg.label] = x
    elif cfg.data_gen['lom'] == 'cos':
        eta_hat = law_of_motion(df, eta1, eta2, eta3)
        lam_hat = law_of_motion(df, lam1, lam2, lam3)
        for i in range(1, n):
            sigma2 = alpha0 + alpha1 * x[i - 1] ** 2 + beta1 * sigma2
            eta = exp_trafo(eta_hat[i], 2.01, 30)
            eta_mean += eta
            lam = exp_trafo(lam_hat[i], -0.99, 0.99)
            lam_mean += lam
            z = SkSt(eta, lam).rvs(1)
            x[i] = np.sqrt(sigma2) * z
            df[cfg.label] = x

    # return pd.DataFrame(x)
    return df, eta_mean/n, lam_mean/n


if __name__ == '__main__':
    main()
