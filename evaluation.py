import pickle
import matplotlib.pyplot as plt
import numpy as np


def main():
    inputs_garch, labels_garch, intervals_garch = pickle.load(open("outputs/intervals/garch_intervals.pickle", "rb"))
    inputs_pb, labels_pb, intervals_pb = pickle.load(open("outputs/intervals/pinball_intervals.pickle", "rb"))
    inputs_qd, labels_qd, intervals_qd = pickle.load(open("outputs/intervals/quality_driven_intervals.pickle", "rb"))

    # for i in range(len(inputs_garch)):
    #     plot_interval(inputs_garch[i], labels_garch[i], intervals_garch[i, :], 'garch', 'r')
    #     plot_interval(inputs_pb[i], labels_pb[i], intervals_pb[i, :], 'pb', 'g')
    #     plot_interval(inputs_qd[i], labels_qd[i], intervals_qd[i, :], 'qd', 'o')
    #     plt.show()

    print_mean_stats(labels_garch, intervals_garch, 'garch')
    print_mean_stats(labels_pb, intervals_pb, 'pb')
    print_mean_stats(labels_qd, intervals_qd, 'qd')


def plot_interval(input, label, interval, name, color):
    plt.plot(input, label=name)
    plt.plot(6, label, 'ko')
    plt.plot(6, interval[0], 'x', color=color, label=name+'_lb')
    plt.plot(6, interval[1], 'x', color=color, label=name+'_ub')
    plt.legend()
    # plt.title('Interval width= ', interval[1]-interval[0])
    return


def print_mean_stats(labels, intervals, name):
    labels = labels.reshape(len(labels))
    K_l = intervals[:, 0] < labels
    K_u = intervals[:, 1] > labels

    print('#############  '+name+'  ###############')
    print('PICP:', np.mean(K_u * K_l))
    print('MPIW:', np.round(np.mean(intervals[:, 1] - intervals[:, 0]), 3))


if __name__ == '__main__':
    main()
