import matplotlib.pyplot as plt
import numpy as np
import config as cfg


def plot_intervals(model, idx, input_data=None, label=None):
    if input_data is not None:
        plt.plot(input_data[idx, -cfg.plot['previous_vals']:], label='Input data')
    if label is not None:
        plt.plot(cfg.plot['previous_vals'], label[idx], 'ko', label='True')
    plt.plot((cfg.plot['previous_vals'], cfg.plot['previous_vals']),
             (model['intervals'][idx, 0], model['intervals'][idx, 1]),
             'x', color=model['plotting']['color'], label=model['name'])
    plt.legend()
    # plt.title('Interval width= ', interval[1]-interval[0])
    return plt


def print_mean_stats(model):
    labels = model['labels'].reshape(len(model['labels']))
    c_l = model['intervals'][:, 0] < labels
    c_u = model['intervals'][:, 1] > labels
    captured = c_u * c_l
    mpiw = np.round(np.mean(model['intervals'][:, 1] - model['intervals'][:, 0]), 3)
    mpiw_c = np.round(np.sum((model['intervals'][:, 1] - model['intervals'][:, 0])*captured)/np.sum(captured), 3)

    print('#############  '+model['name']+'  ###############')
    print('PICP:', np.mean(captured))
    print('MPIW:', mpiw)
    print('MPIW_c: ', mpiw_c)
