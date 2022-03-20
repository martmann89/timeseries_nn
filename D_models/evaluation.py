import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import config as cfg
import utility


def main():
    filename = 'tv_param_cos_2000'
    df = utility.load_df('outputs/' + cfg.data['type'] + '/' + filename + '.pckl')
    eval_param_fit(df, filename)
    plot_hist(df)
    plot_convergence(df)


def eval_param_fit(df, filename):
    """
    evaluates the goodness of distribution models on generated data with a Monte-Carlo approach or real world data

    Parameters
    ----------
    df : DataFrame
        Fitted parameters for monte-carlo experiment or real world data

    Returns
    -------
    None, saves calculated error's to excel
    """
    ### Monte Carlo
    # cols = ['true_parameter', 'avg_est', 'avg_se', 'avg_r_se', 'se_est']
    ### Real World
    cols = ['avg_est', 'avg_se', 'avg_r_se', 'se_est']
    ### TV GARCH
    index = ['alpha0', 'alpha1', 'beta1', 'eta1', 'eta2', 'eta3', 'lam1', 'lam2', 'lam3']
    ### GARCH
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
                arr = np.hstack([cfg.data_gen[param],
                                np.array(df_mean[df_mean.index.str.contains('_' + param + '|^' + param + '$')]),
                                df[param].std()])
        elif len(cols) == 4:
            arr = np.hstack([np.array(df_mean[df_mean.index.str.contains('_' + param + '|^' + param + '$')]),
                             df[param].std()])
        else:
            print('ERROR: length of cols not supported')
        return arr

    for param in index:
        df_eval.loc[param] = build_array(param)
    df_eval['llh'] = df_mean['llh']
    if len(cols) == 4 and len(index) == 9:
        df_eval['eta_mean'] = df_mean['eta']
        df_eval['lam_mean'] = df_mean['lam']
    print(df_eval.head())
    with pd.ExcelWriter('outputs/' + cfg.data['type'] + '/eval_' + filename + '.xlsx') as writer:
        df_eval.to_excel(writer)


def plot_hist(df):
    plt.rcParams['text.usetex'] = True
    sub_plot_x = 3
    ### TV GARCH
    sub_plot_y = 3
    fig, ax = plt.subplots(sub_plot_y, sub_plot_x, figsize=(8, 8))
    ### GARCH
    # sub_plot_y = 2
    # fig, ax = plt.subplots(sub_plot_y, sub_plot_x, figsize=(8, 5))

    ### TV GARCH
    cols = ['alpha0', 'alpha1', 'beta1', 'eta1', 'eta2', 'eta3', 'lam1', 'lam2', 'lam3']
    title = [r'$\alpha_0$', r'$\alpha_1$', r'$\beta_1$', r'$\eta_1$', r'$\eta_2$', r'$\eta_3$',
             r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$']
    ### GARCH
    # cols = ['alpha0', 'alpha1', 'beta1', 'eta', 'lam', 'eta_mean', 'lam_mean', 'llh']
    # title = [r'$\alpha_0$', r'$\alpha_1$', r'$\beta_1$', r'$\eta$', r'$\lambda$', None]

    m = 0
    for i in range(sub_plot_y):
        for j in range(sub_plot_x):
            df.hist(column=cols[m], bins=15, ax=ax[i, j], color='C0', grid=False)
            ax[i, j].set_title(title[m])
            ### TV GARCH
            ax[i, j].vlines(cfg.data_gen[cols[m]], ax[i, j].get_ylim()[0], ax[i, j].get_ylim()[1], color='r')
            ### GARCH
            # if i == 0:
            #     ax[i, j].vlines(cfg.data_gen[cols[m]], ax[i, j].get_ylim()[0], ax[i, j].get_ylim()[1], color='r')
            # else:
            #     ax[i, j].vlines(df[cols[m+2]].loc[0], ax[i, j].get_ylim()[0], ax[i, j].get_ylim()[1], color='r')
            m += 1

    plt.tight_layout()
    ### GARCH
    # ax[1, 2].set_visible(False)
    # ax[1, 0].set_position([0.24, 0.125, 0.228, 0.343])
    # ax[1, 1].set_position([0.55, 0.125, 0.228, 0.343])

    ### plot single vars:
    # df[['alpha1', 'beta1', 'eta2', 'lam3']].plot.hist(subplots=True, bins=50, layout=(2, 2), color='C0',
    #                                                   sharex=False, sharey=False)

    plt.show()


def plot_convergence(df):
    plt.rcParams['text.usetex'] = True
    sub_plot_x = 3
    ### TV GARCH
    sub_plot_y = 3
    fig, ax = plt.subplots(sub_plot_y, sub_plot_x, figsize=(8, 8))
    ### GARCH
    # sub_plot_y = 2
    # fig, ax = plt.subplots(sub_plot_y, sub_plot_x, figsize=(8, 5))
    ### TV GARCH
    cols = ['alpha0', 'alpha1', 'beta1', 'eta1', 'eta2', 'eta3', 'lam1', 'lam2', 'lam3']
    title = [r'$\alpha_0$', r'$\alpha_1$', r'$\beta_1$', r'$\eta_1$', r'$\eta_2$', r'$\eta_3$',
             r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$']
    ### GARCH
    # cols = ['alpha0', 'alpha1', 'beta1', 'eta', 'lam', 'eta_mean', 'lam_mean', 'llh']
    # title = [r'$\alpha_0$', r'$\alpha_1$', r'$\beta_1$', r'$\eta$', r'$\lambda$', None]

    m = 0
    for i in range(sub_plot_y):
        for j in range(sub_plot_x):
            avg = [np.mean(df[cols[m]][0:l]) for l in range(10, len(df)+1, 10)]
            ax[i, j].plot(range(10, len(df)+1, 10), avg)
            ax[i, j].set_title(title[m])
            ### TV GARCH
            ax[i, j].hlines(cfg.data_gen[cols[m]], 10, len(df)+1, 'r')
            ### GARCH
            # if i == 0:
            #     ax[i, j].hlines(cfg.data_gen[cols[m]], 10, len(df)+1, color='r')
            # else:
            #     ax[i, j].hlines(df[cols[m+2]].loc[0], 10, len(df)+1, color='r')
            m += 1
    plt.tight_layout()
    ### GARCH
    # ax[1, 2].set_visible(False)
    # ax[1, 0].set_position([0.24, 0.125, 0.228, 0.343])
    # ax[1, 1].set_position([0.55, 0.125, 0.228, 0.343])

    plt.show()


if __name__ == '__main__':
    main()
