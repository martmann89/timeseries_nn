import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import pylab

config = dict(
    city='ulm',
    var='glo',
    var_name='dayly max glo',
    period=['01.01.2018', '31.12.2020'],  # for data plot
)


def main():
    city = 'ulm'
    # excel2pickle(city)
    df = pd.read_pickle('data/pickles/' + city + '.pickle')
    # print(df.columns)
    # print(df.head())
    df_agg = aggregate_vals(df)
    # df_agg_max = aggregate_vals_max(df)
    # plot_data(df_agg_max)
    # df_agg_max['err'] = df_agg_max['glo'].diff()
    # df_agg_max['err'].plot()
    # plt.show()

    # print(df_agg.columns)
    # print(df_agg.head())
    df = df.set_index('date')
    df_agg['err'] = df_agg['glo'].diff()
    # plot_skew(df_agg, 'err')
    plot_skew(df_agg, 'glo')
    # plot_hist(df_agg, 'glo')
    # df_agg_month = aggregate_months(df_agg)
    # horiz_line = np.array([df_agg['err'].mean() for i in range(len(df_agg_month['err']))])
    # plt.plot(df_agg_month['err'].index, horiz_line)
    # df_agg_month['err'].plot()
    # plt.show()
    # print(df_agg.head(24))
    # df_agg['err'].plot()
    # df_agg['normed'] = df_agg['err']/df_agg['err'].std()
    # plot_hist(df_agg, 'err')
    # df = df.set_index('date')
    # df['err'] = df['glo'].diff(24)
    # df['glo'].plot()
    # plt.show()
    # df['err'].plot()
    # plt.show()
    # plot_hist(df, 'err')
    # plot_skew(df, 'err')
    # df_agg['normed'].plot()
    # plt.show()


def main1():
    city = 'ulm'
    # excel2pickle(city)
    df = pd.read_pickle('data/pickles/' + city + '.pickle')

    df_err = pd.read_pickle('data/pickles/' + city + '_errors.pickle')
    df_err['sum'] = df_err.drop('date', axis=1).sum(axis=1)
    df_err['date'] = pd.to_datetime(df_err['date'], unit='s')
    df_err = df_err.set_index('date')
    plot_hist(df_err, 'sum')

    df = calculate_returns(df)
    # plot_data(df)
    print('skewness of log returns (unaggregated): ', stats.skew(df['log_returns']))

    df_agg = aggregate_vals(df)
    df_agg = calculate_returns(df_agg)
    print('skewness of log returns (aggregated): ', stats.skew(df_agg['log_returns']))
    print('kurtosis of log returns (aggregated): ', stats.kurtosis(df_agg['log_returns']))
    print('jarque_bera of log returns (aggregated): ', stats.jarque_bera(df_agg['log_returns']))
    stats.probplot(df_agg['log_returns'],  dist="norm", plot=pylab)
    # pylab.show()
    # plot_agg_data(df_agg)
    # print_hist(df_agg, 'log_returns')


def plot_data(df):
    df.plot(kind='line', y=config['var'], xlim=config['period'],
            xlabel='', ylabel=config['var_name'],
            figsize=[8, 4],
            legend=None, grid=True)
    plt.show()


def plot_agg_data(df):
    df.plot(kind='line', use_index=True, y=config['var'], xlim=config['period'],
            xlabel='', ylabel=config['var_name'],
            figsize=[8, 4],
            legend=None, grid=True)
    plt.show()


def excel2pickle(city):
    file_location = 'data/' + city + '.xlsx'
    sheet_name = 'data'
    # columns = ['date', 'glo']
    pickle_name = 'data/pickles/' + city + '.pickle'
    df = import_excel(file_location, sheet_name)
    print(df.columns)
    print(df.head())
    df.to_pickle(pickle_name)


def import_excel(file_loc, sheet_name=None, columns=None):
    # data = pd.read_excel(file_loc, sheet_name=sheet_name, index_col='date', usecols=columns)
    data = pd.read_excel(file_loc, sheet_name=sheet_name, usecols=columns, index_col=0)
    return data


def aggregate_vals(df):
    return df.groupby(pd.Grouper(key='date', freq='d')).sum()


def aggregate_vals_max(df):
    return df.groupby(pd.Grouper(key='date', freq='d')).max()


def aggregate_months(df):
    return df.groupby(pd.Grouper(level=0, freq='M')).mean()


def plot_hist(df, var):
    for i in range(1, 13):
        plt.hist(df[var].loc[df.index.month == i], bins=20)
        plt.title(i)
        plt.xlabel('error')
        plt.ylabel('frequency')
        plt.show()


def plot_skew(df, var):
    monthly_skew = []
    # df = df.loc[df.index.year == 2018]
    for i in range(1, 13):
        monthly_skew.append(stats.skew(df[var].loc[df.index.month == i]))
    plt.plot(monthly_skew, '-x')
    plt.show()


def calculate_returns(df):
    df.loc[df['glo'] < 1, ['glo']] = 0
    # df.loc[(df['glo'] < 0.01) & (df['glo'] > 0.0)]['glo'] = 0
    df['returns'] = df['glo'].pct_change()
    df['log_returns'] = np.log(df.glo) - np.log(df.glo.shift(1))
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df['log_returns'] = df['log_returns'].fillna(0)
    df['returns'] = df['returns'].fillna(0)
    return df


if __name__ == '__main__':
    main()
