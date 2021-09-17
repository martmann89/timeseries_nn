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
    # cols = ['true_parameter', 'avg_est', 'avg_se', 'avg_r_se', 'se_est']
    cols = ['avg_est', 'avg_se', 'avg_r_se', 'se_est']
    index = ['alpha0', 'alpha1', 'beta1', 'eta1', 'eta2', 'eta3', 'lam1', 'lam2', 'lam3']
    # index = ['alpha0', 'alpha1', 'beta1', 'eta', 'lam']
    df_mean = df.mean()
    df_eval = pd.DataFrame(columns=cols, index=index)

    def build_array(param):
        arr = None
        if len(cols) == 5:
            if param in ['eta', 'lam']:
                arr = np.hstack([df_mean[param+'_mean'],
                                np.array(df_mean[df_mean.index.str.contains('_'+param+'|^'+param+'$')]),
                                df[param].std()])
            else:
                np.hstack([cfg.data_gen[param],
                           np.array(df_mean[df_mean.index.str.contains('_' + param + '|^' + param + '$')]),
                           df[param].std()])
        elif len(cols) == 4:
            arr = np.hstack([np.array(df_mean[df_mean.index.str.contains('_' + param + '|^' + param + '$')]),
                             df[param].std()])
        else:
            print('SIMON ERROR: length of cols not supported')
        return arr

    for param in index:
        df_eval.loc[param] = build_array(param)
    print(df_eval.head())
    with pd.ExcelWriter('outputs/real_world/tv_garch_param_fit.xlsx') as writer:
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
    captured = utility.calc_capt(model['labels'], model['intervals'])
    mpiw = np.round(np.mean(model['intervals'][:, 1] - model['intervals'][:, 0]), 3)
    mpiw_c = np.round(np.sum((model['intervals'][:, 1] - model['intervals'][:, 0])*captured)/np.sum(captured), 3)

    print('#############  '+model['name']+'  ###############')
    print('PICP:', np.mean(captured))
    print('MPIW:', mpiw)
    print('MPIW_c: ', mpiw_c)


if __name__ == '__main__':
    main()
