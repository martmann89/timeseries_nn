from scipy.optimize import minimize
import numpy as np
import pandas as pd
from operator import mod
# import matplotlib.pyplot as plt
# from scipy.stats import t
from statsmodels.tools.numdiff import approx_fprime, approx_hess

import arch.data.sp500
from D_models.SkewStudent import SkewStudent
import config as cfg
import config_models as cfg_mod
from utility import exp_trafo, law_of_motion
# from evaluation import plot_intervals

import time

alpha_ = cfg.prediction['alpha']
# alpha_ = 0.025

SIGMA = None


def run_single_garch_tv(data_set):
    bounds = cfg_mod.model_garch_tv['bounds']
    starting_values = cfg_mod.model_garch_tv['starting_values']

    kwargs = {'resids': data_set, 'individual': False}
    # cons = {'type': 'eq', 'fun': lambda x: x[5] - x[8]}
    # start = time.process_time()
    res = minimize(loglikelihood, starting_values, args=kwargs['resids'], method='SLSQP',
                   bounds=bounds,
                   # constraints=cons,
                   )
    # ende = time.process_time()
    params, llh = _eval_opt(res), (-1) * res.fun
    # print(params)
    # print(llh)
    # print('Minimierung: {:5.3f}s'.format(ende - start))
    # std_err = calc_se(loglikelihood, np.array([*params.values()]), kwargs, robust=False)
    # start = time.process_time()
    std_err, robust_std_err = calc_se(loglikelihood, np.array([*params.values()]), kwargs, robust=True)
    # ende = time.process_time()
    # print('Fehlerberchnung: {:5.3f}s'.format(ende - start))

    # print(std_err)
    # std_err = robust_std_err = None
    return params, llh, std_err, robust_std_err


def run_garch_tv(data_set, m_storage):
    input_len = cfg.d_pred['input_len']  # not validated yet
    # input_len = 1000
    # test_len = cfg.data['test_data_size'] + cfg.nn_pred['input_len']
    test_len = cfg.data['test_data_size']
    # test_len = 1

    intervals = np.empty((0, 2), float)
    labels = np.empty((0, 1), float)
    inputs = np.empty((0, input_len, 1), float)
    params = list()
    etas = list()
    lams = list()
    classic_se = list()
    robust_se = list()
    llh = list()

    for i in range(len(data_set)-test_len, len(data_set)):
        print(i-(len(data_set)-test_len))
        train, true = _get_traindata(input_len, data_set, i)
        # cons = {'type': 'ineq', 'fun': lambda x: -(x[1] + x[2])+1}
        # cons = {'type': 'eq', 'fun': lambda x: x[5]-x[8]}
        kwargs = {'resids': train, 'individual': False}
        res = minimize(loglikelihood, m_storage['starting_values'], args=kwargs['resids'], method='SLSQP',
                       bounds=m_storage['bounds'],
                       # constraints=cons,
                       )
        param = _eval_opt(res)
        llh.append((-1) * res.fun)
        std_err, robust_std_err = calc_se(loglikelihood, np.array([*param.values()]), kwargs, robust=True)

        sigma, eta, lam = forecast(train, param.values())
        params.append(list(param.values()))
        etas.append(eta)
        lams.append(lam)
        classic_se.append(std_err)
        robust_se.append(robust_std_err)
        intervals = np.append(intervals, _get_interval(true, (sigma, eta, lam)), axis=0)
        labels = np.append(labels, np.array(true[cfg.label]).reshape(1, 1), axis=0)
        inputs = np.append(inputs, np.array(train[cfg.label]).reshape((1, input_len, 1)), axis=0)

    m_storage['intervals'] = intervals
    m_storage['inputs'] = inputs
    m_storage['labels'] = labels
    m_storage['etas'] = etas
    m_storage['lams'] = lams
    m_storage['params'] = np.array(params)
    m_storage['classic_se'] = np.array(classic_se)
    m_storage['robust_se'] = np.array(robust_se)
    m_storage['LogLikelihood'] = llh
    return m_storage


def _get_traindata(length, df, true_idx):
    true_val = df.iloc[[true_idx]]
    # train_data = data[cfg.label][true_idx-length:true_idx].reset_index(drop=True)
    # train_data = pd.DataFrame(df[cfg.label][true_idx-length:true_idx], columns=[cfg.label])
    train_data = df[true_idx-length:true_idx].reset_index(drop=True)
    return train_data, true_val


def _get_interval(true_val, comb_param, print_results=False):
    sigma, eta, lam = comb_param
    skewt_dist = SkewStudent(eta=eta, lam=lam)
    lb, ub = skewt_dist.ppf(alpha_ / 2), skewt_dist.ppf(1 - alpha_ / 2)
    interval = np.array([[sigma * lb, sigma * ub]])
    if print_results:
        print('True value: ', true_val)
        print(f'Parameter - dof={eta}; lambda={lam}')
        print(f'lower bound quantile = {lb}, upper bound quantile = {ub}')
        print('Standard deviation =', sigma)
        print(f'Interval: {interval}, width=', interval[1] - interval[0])
    return interval


def forecast(x, param):
    a0, a1, b, eta1, eta2, eta3, lam1, lam2, lam3 = param
    # sigma = garch_filter(x[cfg.label], a0, a1, b, np.var(x[cfg.label]))[-1]
    sigma = np.sqrt(a0 + a1 * x[cfg.label].iloc[-1]**2 + b * SIGMA)
    # sigma = np.sqrt(sigma_2)
    # x_val = x[cfg.label].iloc[-1]
    # eta = exp_trafo(eta1 + eta2*x_val + eta3*x_val**2, 2.1, 30)
    # lam = exp_trafo(lam1 + lam2*x_val + lam3*x_val**2, -0.99, 0.99)
    x_val = mod(x['#day'].iloc[-1]+1, 365)
    # TODO: implement correct LoM
    eta = exp_trafo(eta1 + eta2*np.cos(((x_val-eta3)/365)*2*np.pi), 2.1, 30)
    lam = exp_trafo(lam1 + lam2*np.cos(((x_val-lam3)/365)*2*np.pi), -0.99, 0.99)
    return sigma, eta, lam


def loglikelihood(param, resids=None, individual=False):
    len_par = len(param)
    sigma, eta, lam = None, None, None
    if len_par == 9:  # GARCH(1,1) with tv eta and lambda
        a0, a1, b, eta1, eta2, eta3, lam1, lam2, lam3 = param
        lam_hat = law_of_motion(resids, lam1, lam2, lam3)
        # if any(lam_hat < -500):
        #     print('SIMON: lam kleiner -500')
        lam = exp_trafo(lam_hat, -0.99, 0.99)
        eta_hat = law_of_motion(resids, eta1, eta2, eta3)
        # if any(eta_hat < -500):
        #     print('SIMON: eta kleiner -500')
        eta = exp_trafo(eta_hat, 2.1, 30)
        sv_garch = np.var(resids['d_glo'])
        sigma = garch_filter(np.array(resids['d_glo']), a0, a1, b, sv_garch)
    if len_par == 7:  # GARCH(1,1) with tv eta
        a0, a1, b, eta1, eta2, eta3, lam = param
        eta_hat = law_of_motion(resids, eta1, eta2, eta3)
        eta = exp_trafo(eta_hat, 2.1, 30)
        sv_garch = np.var(resids)
        sigma = garch_filter(resids, a0, a1, b, sv_garch)
    if len_par == 5:  # GARCH(1,1)
        a0, a1, b, eta, lam = param
        sv_garch = np.var(resids)
        sigma = garch_filter(resids, a0, a1, b, sv_garch)
    if len_par == 4:  # GARCH(1,0)
        a0, a1, eta, lam = param
        sigma = arch_filter(resids, a0, a1)
    global SIGMA
    SIGMA = sigma[-1]

    if individual:
        return (-1)*loglike_innovations(resids[cfg.label].iloc[1:]/sigma, eta, lam) - np.log(sigma)
    else:
        return -1*np.sum(loglike_innovations(resids[cfg.label].iloc[1:]/sigma, eta, lam) - np.log(sigma))


def arch_filter(x, a0, a1):
    it = len(x)+1
    sigma_2 = np.zeros(it)
    for i in range(1, it):
        sigma_2[i] = a0 + a1*x[i-1]**2
    return np.sqrt(sigma_2[1:])


def garch_filter(x, a0, a1, b, sv):
    it = len(x)
    sigma_2 = np.zeros(it)
    sigma_2[0] = sv
    for i in range(1, it):
        sigma_2[i] = a0 + a1*x[i-1]**2 + b*sigma_2[i-1]
    return np.sqrt(sigma_2[1:])


def loglike_innovations(z, eta, lam):
    return SkewStudent().loglikelihood((eta, lam), z)
    # return t.logpdf(z, eta, loc=0, scale=1)


def _eval_opt(res):
    params = dict(
        alpha0=res.x[0],
        alpha1=res.x[1],
        beta1=res.x[2],
        eta1=res.x[3],
        eta2=res.x[4],
        eta3=res.x[5],
        lam1=res.x[6],
        lam2=res.x[7],
        lam3=res.x[8],
    )
    return params


def calc_se(f, params, kwargs, robust=False):
    nobs = len(kwargs['resids'])
    hess = approx_hess(params, f, kwargs=kwargs)
    hess /= nobs
    inv_hess = np.linalg.pinv(hess)
    param_cov = inv_hess / nobs
    if robust:
        kwargs["individual"] = True
        scores = approx_fprime(
            params, f, kwargs=kwargs
        )  # type: np.ndarray
        score_cov = np.cov(scores.T)
        robust_param_cov = inv_hess.dot(score_cov).dot(inv_hess) / nobs
        diag = np.diag(param_cov).copy()
        diag[diag < 0] = 0
        diag_robust = np.diag(robust_param_cov).copy()
        diag_robust[diag_robust < 0] = 0
        return np.sqrt(diag), np.sqrt(diag_robust)
    return np.sqrt(np.diag(param_cov))

# def eval_opt(model, res):
#     model['LLH'] = (-1)*res.fun
#     model['param']['alpha0'] = res.x[0]
#     model['param']['alpha1'] = res.x[1]
#     model['param']['beta1'] = res.x[2]
#     model['param']['eta1'] = res.x[3]
#     model['param']['eta2'] = res.x[4]
#     model['param']['eta3'] = res.x[5]
#     model['param']['lam1'] = res.x[6]
#     model['param']['lam2'] = res.x[7]
#     model['param']['lam3'] = res.x[8]
#     return model


if __name__ == '__main__':
    # data = arch.data.sp500.load()
    # market = pd.DataFrame(data, columns={"Adj Close"})
    # market = market.rename(columns={'Adj Close': 'd_glo'})
    # resids = market.diff(1).dropna()[:1000]
    file_loc = 'data/pickles/time_varying_data_2230.pckl'
    data = pd.read_pickle(file_loc)
    df_ = pd.DataFrame(data, columns=['d_glo'])

    model = cfg_mod.model_garch_tv

    ### reset index of df
    df_['#day'] = list(map(lambda x: mod(x, 365), range(len(df_))))

    # store = run_single_garch_tv(df_)
    model = run_garch_tv(df_, model)
    print('Hello World')
