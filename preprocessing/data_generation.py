import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import mean_squared_error as mse

# from NN_models.qd_loss import qd_objective
from D_models.SkewStudent import SkewStudent as SkSt

from D_models.garch_tv import run_single_garch_tv
from D_models.garch import run_single_garch
from utility import exp_trafo, law_of_motion

import config as cfg


def main():
    filepath = 'outputs/monte_carlo/garch_tv_param_1000.pckl'
    # filepath = 'data/pickles/time_varying_data_1500.pckl'
    # n_mc = cfg.monte_carlo
    # n_mc = 200
    # data = eval_garch_tv(n_mc)
    # print('Hallo')
    # data, _, _ = single_ts_generation(0, 1, 1865)
    # plt.plot(data)
    # plt.show()
    # save_df(data, filepath)
    with open(filepath, 'rb') as file_scaler:
        df = pickle.load(file_scaler)
    # test = eval_garch_tv(1)
    df[['alpha1', 'beta1', 'eta1', 'lam1']].plot.hist(subplots=True, bins=50, layout=(2, 2), color='C0')
    plt.show()
    # print(df[['lam1', 'lam2', 'lam3']].mean())
    # print(mse(np.array([np.full(n_mc, b1), np.full(n_mc, b2), np.full(n_mc, b3)]).transpose(),
    #           df[['lam1', 'lam2', 'lam3']],
    #           multioutput='raw_values'))
    # x0 = 0
    # sigma0 = 1
    # n = 1500
    # x = single_ts_generation(x0, sigma0, n)
    # plt.plot(x)
    # plt.show()
    # params, llh = run_single_garch_tv(x)


def save_df(df, filepath):
    file_scaler = open(filepath, 'wb')
    pickle.dump(df, file_scaler)


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
        x, _, _ = single_ts_generation(x0, sigma0, n)
        param, llh, std_err, robust_std_err = run_single_garch_tv(x)
        full_dict = {**param,
                     **dict(zip(llh_col, [llh])),
                     **dict(zip(se_cols, std_err)),
                     **dict(zip(r_se_cols, robust_std_err))
                     }
        df = df.append(full_dict, ignore_index=True)
    return df


def single_ts_generation(x0, sigma0, n):
    x = np.zeros(n)
    x[0] = x0
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
    for i in range(1, n):
        sigma2 = alpha0 + alpha1 * x[i - 1]**2 + beta1 * sigma2
        eta_hat = eta1 + eta2 * x[i - 1] + eta3 * x[i - 1] ** 2
        # eta_hat = law_of_motion(x[i-1], eta1, eta2, eta3)
        eta = exp_trafo(eta_hat, 2.01, 30)
        eta_mean += eta
        lam_hat = lam1 + lam2 * x[i - 1] + lam3 * x[i - 1] ** 2
        # lam_hat = law_of_motion(x[i-1], lam1, lam2, lam3)
        lam = exp_trafo(lam_hat, -0.99, 0.99)
        lam_mean += lam
        # print((eta, lam))
        z = SkSt(eta, lam).rvs(1)
        x[i] = np.sqrt(sigma2) * z
    # return pd.DataFrame(x)
    return x, eta_mean/n, lam_mean/n


if __name__ == '__main__':
    main()
