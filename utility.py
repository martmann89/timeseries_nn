import numpy as np
import pickle


def exp_trafo(param, lb, ub):
    return lb + (ub-lb)/(1+np.exp(-param))


def law_of_motion(x, par1, par2, par3):
    it = len(x)
    par_hat = np.zeros(it)
    par_hat[0] = 0
    for i in range(1, it):
        par_hat[i] = par1 + par2*x[i - 1] + par3*x[i - 1]**2
    return par_hat[1:]


def load_df(path):
    with open(path, 'rb') as file_scaler:
        return pickle.load(file_scaler)
