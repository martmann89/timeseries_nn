import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

import arch.data.sp500

import models.model_handling as m_handling
from preprocessing.WindowGenerator import WindowGenerator
import preprocessing.preprocessing as pp
import config as cfg
import config_models as cfg_mod


def main():
    # get and preprocess data
    data = arch.data.sp500.load()
    market = pd.DataFrame(data, columns={"Adj Close"})
    market = market.rename(columns={'Adj Close': 'd_glo'})
    df = market.diff(1).dropna()
    m_storage = cfg_mod.pb_model
    train, val, test = pp.preprocess(df, [cfg.label])
    m_storage['train'], m_storage['val'], m_storage['test'] = train, val, test

    with open('./outputs/scaler/output_scaler.pckl', 'rb') as file_scaler:
        m_storage['scaler'] = pickle.load(file_scaler)

    m_storage['window'] = WindowGenerator(input_width=cfg.prediction['input_len'],
                                          label_width=cfg.prediction['num_predictions'],
                                          train_df=train, val_df=val, test_df=test,
                                          shift=cfg.prediction['num_predictions'],
                                          label_columns=[cfg.label],
                                          )

    m_storage['model'] = m_handling.build_model(m_handling.choose_model_loss(m_storage),
                                                m_storage['window'],
                                                './checkpoints/test_data/' + m_storage['loss'],
                                                train=m_storage['train_bool'])

    def plot_interval(input, label, interval, scaler):
        input = scaler.inverse_transform(input)
        label = scaler.inverse_transform(label.reshape(-1, 1))
        plt.plot(input)
        plt.plot(6, label, 'ko')
        plt.plot(6, interval[0], 'gx')
        plt.plot(6, interval[1], 'rx')
        plt.show()

    def scale_data_back(data, scaler):
        if len(data.shape) == 3:
            d1, d2, d3 = data.shape
            data = data.reshape(d1*d2, d3)
            data = scaler.inverse_transform(data)
            return data.reshape(d1, d2, d3)
        else:
            return scaler.inverse_transform(data)

    intervals_store = np.empty((0, 2), float)
    labels_store = np.empty((0, 1), float)
    inputs_store = np.empty((0, 6, 1), float)
    for data in m_storage['window'].test:
        inputs, labels = data
        intervals = m_handling.get_predictions(m_storage['model'], inputs, m_storage['scaler'])
        intervals_store = np.append(intervals_store, intervals, axis=0)
        labels_store = np.append(labels_store, np.array(labels[:, 0]), axis=0)
        inputs_store = np.append(inputs_store, np.array(inputs), axis=0)
        # interval = np.concatenate(y_pred, axis=1)
        # for i in range(len(inputs)):
        #     plot_interval(inputs.numpy()[i], labels.numpy()[i, 0], y_pred[i, :], m_storage['scaler'])

    with open('outputs/intervals/' + m_storage['loss'] + '_intervals.pickle', 'wb') as f:
        pickle.dump([scale_data_back(inputs_store, m_storage['scaler']),
                     scale_data_back(labels_store, m_storage['scaler']),
                     intervals_store], f)
    # view training
    # result_loss = np.array(result_loss).reshape(-1)
    # x = range(result_loss.shape[0])
    # plt.plot(x, result_loss, label='train')
    # plt.title('Loss')
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.ylim(0, result_loss.max())
    # plt.show()

    # print some stats
    # y_pred = model.predict(x_test, verbose=0)
    # if type_of_loss == 'qd':
    #     y_u_pred = y_pred[:, 0]
    #     y_l_pred = y_pred[:, 1]
    #     K_u = y_u_pred > y_test
    #     K_l = y_l_pred < y_test
    # elif type_of_loss == 'pinball':
    #     y_u_pred = y_pred[0].reshape(-1)
    #     y_l_pred = y_pred[1].reshape(-1)
    #     K_u = y_u_pred > y_test
    #     K_l = y_l_pred < y_test

    # print('PICP:', np.mean(K_u * K_l))
    # print('MPIW:', np.round(np.mean(y_u_pred - y_l_pred), 3))


if __name__ == '__main__':
    main()
