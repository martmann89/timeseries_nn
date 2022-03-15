from scipy.optimize import minimize
import numpy as np
import pandas as pd
from operator import mod
from statsmodels.tools.numdiff import approx_fprime, approx_hess
import arch.data.sp500

from D_models.SkewStudent import SkewStudent
import config as cfg
import config_models as cfg_mod
from utility import exp_trafo, law_of_motion

alpha_ = cfg.prediction['alpha']
### alpha for conformal prediction
# alpha_ = 0.08
SIGMA = None


def run_garch_tv(data_set, m_storage):
    """
    Returns dictionary with prediction intervals (at given confidence level)
     calculated with TV-GARCH(1,1) with Hansen's skewed t distribution.
     Moreover the inputs which were used for parameter calibration
     and the true value of the predicted interval are returned.
     Parameters lambda and eta are returned.

    Parameters
    ----------
    data_set : DataFrame
        input data set for parameter fitting
    m_storage : dict
        dictionary with information of used model, defined in "config_models.py"-file

    Returns
    -------
    m_storage: dict
        intervals, inputs, labels, parameter, standard errors and llh added
    """
    input_len = cfg.d_pred['input_len']  # not validated yet
    test_len = cfg.data['test_data_size']

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
        kwargs = {'resids': train, 'individual': False}
        res = minimize(loglikelihood, m_storage['starting_values'], args=kwargs['resids'], method='SLSQP',
                       bounds=m_storage['bounds'],
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
        intervals = np.append(intervals, _get_interval((sigma, eta, lam)), axis=0)
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


def run_single_garch_tv(data_set):
    """
    Returns dictionary with prediction intervals (at given confidence level)
     calculated with TV-GARCH(1,1) with Hansen's skewed t distribution.
     Moreover the inputs which were used for parameter calibration
     and the true value of the predicted interval are returned.
     Parameters lambda and eta are returned.

    Parameters
    ----------
    data_set : DataFrame
        input data set for parameter fitting
    Returns
    -------
    params : dict
        parameter of garch fitting

    llh : float
        loglikelihood value of fitted process on data_set

    std_err: float
        standard error of fit

    robust_std_err: float
        robust standard error of fit
    """
    bounds = cfg_mod.model_garch_tv_q['bounds']
    starting_values = cfg_mod.model_garch_tv_q['starting_values']

    kwargs = {'resids': data_set, 'individual': False}
    res = minimize(loglikelihood, starting_values, args=kwargs['resids'], method='SLSQP',
                   bounds=bounds,
                   )
    params, llh = _eval_opt(res), (-1) * res.fun
    std_err, robust_std_err = calc_se(loglikelihood, np.array([*params.values()]), kwargs, robust=True)

    return params, llh, std_err, robust_std_err


def _get_traindata(length, df, true_idx):
    true_val = df.iloc[[true_idx]]
    train_data = df[true_idx-length:true_idx].reset_index(drop=True)
    return train_data, true_val


def _get_interval(comb_param):
    sigma, eta, lam = comb_param
    skewt_dist = SkewStudent(eta=eta, lam=lam)
    lb, ub = skewt_dist.ppf(alpha_ / 2), skewt_dist.ppf(1 - alpha_ / 2)
    interval = np.array([[sigma * lb, sigma * ub]])
    return interval


def forecast(x, param):
    eta, lam = None, None
    a0, a1, b, eta1, eta2, eta3, lam1, lam2, lam3 = param
    # sigma = garch_filter(x[cfg.label], a0, a1, b, np.var(x[cfg.label]))[-1]
    sigma = np.sqrt(a0 + a1 * x[cfg.label].iloc[-1]**2 + b * SIGMA)
    # sigma = np.sqrt(sigma_2)
    if cfg.data_gen['lom'] == 'quad':
        x_val = x[cfg.label].iloc[-1]
        eta = exp_trafo(eta1 + eta2*x_val + eta3*x_val**2, 2.1, 30)
        lam = exp_trafo(lam1 + lam2*x_val + lam3*x_val**2, -0.99, 0.99)
    if cfg.data_gen['lom'] == 'cos':
        x_val = x['#day'].iloc[-1]+1
        eta = exp_trafo(eta1 + eta2*np.cos(((x_val-eta3)/365)*2*np.pi), 2.1, 30)
        lam = exp_trafo(lam1 + lam2*np.cos(((x_val-lam3)/365)*2*np.pi), -0.99, 0.99)
    return sigma, eta, lam


def loglikelihood(param, resids=None, individual=False):
    a0, a1, b, eta1, eta2, eta3, lam1, lam2, lam3 = param
    lam_hat = law_of_motion(resids, lam1, lam2, lam3)
    lam = exp_trafo(lam_hat, -0.99, 0.99)
    eta_hat = law_of_motion(resids, eta1, eta2, eta3)
    eta = exp_trafo(eta_hat, 2.1, 30)
    sv_garch = np.var(resids['d_glo'])
    sigma = garch_filter(np.array(resids['d_glo']), a0, a1, b, sv_garch)
    global SIGMA
    SIGMA = sigma[-1]
    if individual:
        return (-1)*loglike_innovations(resids[cfg.label].iloc[1:]/sigma, eta, lam) - np.log(sigma)
    else:
        return -1*np.sum(loglike_innovations(resids[cfg.label].iloc[1:]/sigma, eta, lam) - np.log(sigma))


def garch_filter(x, a0, a1, b, sv):
    it = len(x)
    sigma_2 = np.zeros(it)
    sigma_2[0] = sv
    for i in range(1, it):
        sigma_2[i] = a0 + a1*x[i-1]**2 + b*sigma_2[i-1]
    return np.sqrt(sigma_2[1:])


def loglike_innovations(z, eta, lam):
    return SkewStudent().loglikelihood((eta, lam), z)


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


if __name__ == '__main__':
    data = arch.data.sp500.load()
    market = pd.DataFrame(data, columns={"Adj Close"})
    market = market.rename(columns={'Adj Close': 'd_glo'})
    df_ = market.diff(1).dropna()

    # file_loc = 'data/pickles/time_varying_data_2230.pckl'
    # data = pd.read_pickle(file_loc)
    # df_ = pd.DataFrame(data, columns=['d_glo'])

    model = cfg_mod.model_garch_tv_q

    ### reset index of df
    df_['#day'] = list(map(lambda x: mod(x, 365), range(len(df_))))

    # store = run_single_garch_tv(df_)
    model = run_garch_tv(df_, model)
    print('Hello World')
