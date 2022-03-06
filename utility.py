import numpy as np
import pickle
import config as cfg


def exp_trafo(param, lb, ub):
    return lb + (ub-lb)/(1+np.exp(-param))


def law_of_motion(x, par1, par2, par3):
    if cfg.data_gen['lom'] == 'quad':
        x = np.array(x[cfg.label])[:-1]
        par_hat = par1 + par2 * x + par3 * x ** 2
    elif cfg.data_gen['lom'] == 'cos':
        par_hat = par1 + par2*np.cos(((x['#day']-par3)/365)*2*np.pi)[1:]
    else:
        par_hat = None

    # it = len(x)
    # par_hat = np.zeros(it)
    # for i in range(1, it):
    #     par_hat[i] = par1 + par2*x[i - 1] + par3*x[i - 1]**2
    # return par_hat[1:]

    return par_hat


def load_df(path):
    with open(path, 'rb') as file_scaler:
        return pickle.load(file_scaler)


def save_df(df, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(df, f)


def save_model(model, filepath):
    with open(filepath+'.pickle', 'wb') as f:
        pickle.dump(model, f)


def load_model(path):
    with open(path, 'rb') as file_scaler:
        return pickle.load(file_scaler)


def calc_capt(labels, intervals):
    labels = labels.reshape(len(labels))
    c_l = intervals[:, 0] < labels
    c_u = intervals[:, 1] > labels
    return c_u * c_l
