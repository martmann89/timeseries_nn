import arch.data.sp500

from arch import arch_model
import numpy as np
import pandas as pd

from D_models.SkewStudent import SkewStudent
import config as cfg
import config_models as cfg_mod

horizon_ = cfg.prediction['horizon']
alpha_ = cfg.prediction['alpha']
### alpha for conformal prediction
# alpha_ = 0.077


def run_garch(data_set, m_storage):
    """
    Returns dictionary with prediction intervals (at given confidence level)
     calculated with GARCH(1,1) with Hansen's skewed t distribution.
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
    alpha0 = list()
    alpha1 = list()
    beta1 = list()
    eta = list()
    lam = list()
    classic_se = list()
    robust_se = list()
    llh = list()
    for i in range(len(data_set)-test_len, len(data_set)):
        print(i-(len(data_set)-test_len))
        train, true = _get_traindata(input_len, data_set, i)
        am = arch_model(train, p=1, q=1,
                        dist="skewt",
                        mean='zero',
                        )
        res = am.fit(update_freq=0, disp='off', cov_type='classic')
        res_robust = am.fit(update_freq=0, disp='off', cov_type='robust')

        llh.append(res.loglikelihood)
        alpha0.append(res.params['omega'])
        alpha1.append(res.params['alpha[1]'])
        beta1.append(res.params['beta[1]'])
        eta.append(res.params['nu'])
        lam.append(res.params['lambda'])
        classic_se.append(res.std_err)
        robust_se.append(res_robust.std_err)
        intervals = np.append(intervals, _get_interval(res), axis=0)
        labels = np.append(labels, np.array(true).reshape(1, 1), axis=0)
        inputs = np.append(inputs, np.array(train).reshape((1, input_len, 1)), axis=0)

    m_storage['intervals'] = intervals
    m_storage['inputs'] = inputs
    m_storage['labels'] = labels
    m_storage['etas'] = eta
    m_storage['lams'] = lam
    m_storage['alpha0'] = alpha0
    m_storage['alpha1'] = alpha1
    m_storage['beta1'] = beta1
    m_storage['classic_se'] = np.array(classic_se)
    m_storage['robust_se'] = np.array(robust_se)
    m_storage['LogLikelihood'] = llh
    return m_storage


def run_single_garch(data_set):
    """
    Returns dictionary with prediction intervals (at given confidence level)
     calculated with GARCH(1,1) with Hansen's skewed t distribution.
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

    loglikelihood : float
        loglikelihood value of fitted time series

    std_err: float
        (robust) standard error of fit
    """
    am = arch_model(data_set[cfg.label], p=1, q=1,
                    dist="skewt",
                    mean='zero',
                    )
    res_classic = am.fit(update_freq=0, disp='off', cov_type='classic')
    res_robust = am.fit(update_freq=0, disp='off', cov_type='robust')
    params = res_robust.params
    params = dict(alpha0=params['omega'],
                  alpha1=params['alpha[1]'],
                  beta1=params['beta[1]'],
                  eta=params['nu'],
                  lam=params['lambda']
                  )

    return params, res_classic.loglikelihood, res_classic.std_err, res_robust.std_err


def _get_traindata(length, data, true_idx):
    true_val = data[cfg.label][true_idx]
    train_data = data[cfg.label][true_idx-length:true_idx]
    return train_data, true_val


def _get_interval(fit_res):
    eta, lam = fit_res.params['nu'], fit_res.params['lambda']
    skewt_dist = SkewStudent(eta=eta, lam=lam)
    lb, ub = skewt_dist.ppf(alpha_ / 2), skewt_dist.ppf(1 - alpha_ / 2)
    fcast = fit_res.forecast(horizon=horizon_, reindex=False)
    std_dev = np.sqrt(float(fcast.variance['h.1']))
    interval = np.array([[std_dev * lb, std_dev * ub]])
    return interval


if __name__ == '__main__':
    # example dataset
    data = arch.data.sp500.load()
    market = pd.DataFrame(data, columns={"Adj Close"})
    market = market.rename(columns={'Adj Close': 'd_glo'})
    returns = market.diff(1).dropna()
    model = cfg_mod.model_garch
    store = run_single_garch(returns)
