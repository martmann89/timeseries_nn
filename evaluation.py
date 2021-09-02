import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import config as cfg
import utility


def main():
    df = utility.load_df('outputs/monte_carlo/garch_tv_param_1000.pckl')
    eval_param_fit(df)


def eval_param_fit(df):
    """
    evaluates the goodness of distribution models on generated data with a Monte-Carlo approach

    Parameters
    ----------
    df : DataFrame
        Fitted parameters for monte-carlo experiment

    Returns
    -------
    None, saves calculated error's to excel
    """
    cols = ['true_parameter', 'avg_est', 'avg_se', 'avg_r_se', 'se_est']
    index = ['alpha0', 'alpha1', 'beta1', 'eta1', 'eta2', 'eta3', 'lam1', 'lam2', 'lam3']
    # index = ['alpha0', 'alpha1', 'beta1', 'eta', 'lam']
    df_mean = df.mean()
    df_eval = pd.DataFrame(columns=cols, index=index)
    for param in index:
        if param in ['eta', 'lam']:
            df_eval.loc[param] = np.hstack([df_mean[param+'_mean'],
                                           np.array(df_mean[df_mean.index.str.contains('_'+param+'|^'+param+'$')]),
                                           df[param].std()])
        else:
            df_eval.loc[param] = np.hstack([cfg.data_gen[param],
                                           np.array(df_mean[df_mean.index.str.contains('_'+param+'|^'+param+'$')]),
                                           df[param].std()])
    with pd.ExcelWriter('outputs/monte_carlo/tv_param_fit_1000.xlsx') as writer:
        df_eval.to_excel(writer)


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


if __name__ == '__main__':
    main()
