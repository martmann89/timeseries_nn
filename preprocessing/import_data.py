import pandas as pd
import matplotlib.pyplot as plt
import config as cfg


def main():
    # city = 'lemberg'
    file_location = 'data/PV_Daten.xlsx'
    # sheet_name = 'data'
    pickle_name = 'data/pickles/PV_Daten_returns.pickle'
    df = import_excel(file_location, sheet_name='Tabelle1')
    df[cfg.label] = df.sum(axis=1)
    df = df.diff(1).dropna()
    print(df.head())
    df[[cfg.label]].to_pickle(pickle_name)


def import_excel(file_loc, sheet_name=None, columns=None):
    data = pd.read_excel(file_loc, index_col='date', sheet_name=sheet_name)
    if columns is not None:
        data = pd.DataFrame(data, columns=columns)
    return data


def visualize(df, date_time):
    plot_cols = ['glo', 'maxIncoming', 'difference']
    plot_features = df[plot_cols]
    plot_features.index = date_time
    _ = plot_features.plot(subplots=True)

    plot_features = df[plot_cols][:480]
    plot_features.index = date_time[:480]
    _ = plot_features.plot(subplots=True)

    plt.show()


if __name__ == '__main__':
    main()

