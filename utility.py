import numpy as np
import matplotlib.pyplot as plt
import pickle
import config as cfg
import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))



def scale_intervals(model, scaler):
    model['intervals'] = scaler.inverse_transform(model['intervals'])
    model['labels'] = scaler.inverse_transform(model['labels'])
    return model


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


def print_mean_stats(model):
    captured = calc_capt(model['labels'], model['intervals'])
    mpiw = np.round(np.mean(model['intervals'][:, 1] - model['intervals'][:, 0]), 3)
    mpiw_c = np.round(np.sum((model['intervals'][:, 1] - model['intervals'][:, 0])*captured)/np.sum(captured), 3)

    print('#############  '+model['name']+'  ###############')
    print('PICP:', np.mean(captured))
    print('MPIW:', mpiw)
    print('MPIW_c: ', mpiw_c)
    return np.mean(captured), mpiw


def plot_intervals(model, idx, input_data=None, label=None):
    if input_data is not None:
        plt.plot(input_data[idx, -cfg.plot['previous_vals']:], label='Input data')
    if label is not None:
        plt.plot(cfg.plot['previous_vals'], label[idx], 'ko', label='True')
    plt.plot((cfg.plot['previous_vals'], cfg.plot['previous_vals']),
             (model['intervals'][idx, 0], model['intervals'][idx, 1]),
             'x', color=model['plotting']['color'], label=model['name'])
    plt.legend()
    return plt


def plot_intervals2(model, start_idx, end_idx, input_data=None, label=None, date_index=None):
    if input_data is not None:
        plot_data = np.append(input_data[start_idx, -cfg.plot['previous_vals']:], label[start_idx])
        plt.plot(plot_data, label='Input data')
    if label is not None:
        plt.plot(range(cfg.plot['previous_vals'], cfg.plot['previous_vals']+end_idx-start_idx),
                 label[start_idx:end_idx], 'ko--', label='True')
    plt.plot(range(cfg.plot['previous_vals'], cfg.plot['previous_vals']+end_idx-start_idx),
             model['intervals'][start_idx:end_idx, 0],
             ':', color=model['plotting']['color'], marker=model['plotting'].get('marker', 'x'))
    plt.plot(range(cfg.plot['previous_vals'], cfg.plot['previous_vals']+end_idx-start_idx),
             model['intervals'][start_idx:end_idx, 1],
             ':', color=model['plotting']['color'], marker=model['plotting'].get('marker', 'x'), label=model['name'])
    # plt.legend(loc='lower left')
    plt.legend()

    if date_index is not None:
        x_ticks = list(range(0, end_idx-start_idx+cfg.plot['previous_vals'], 2))
        x_tick_label = date_index[start_idx-cfg.plot['previous_vals']:end_idx:2].strftime("%d.%m.%Y")
        plt.xticks(x_ticks, x_tick_label)
        plt.gcf().autofmt_xdate()

    mpiw = np.round(np.mean(model['intervals'][start_idx:end_idx, 1] - model['intervals'][start_idx:end_idx, 0]), 3)
    print('#############  '+model['name']+'  ###############')
    print('MPIW:', mpiw)
    return plt