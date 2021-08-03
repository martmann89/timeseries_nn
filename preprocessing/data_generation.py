import numpy as np
from NN_models.qd_loss import qd_objective

type_of_loss = 'qd'  # 'qd' or 'pinball'
type_of_data = '1'  # '1' or '2'


def main():
    if type_of_data == '1':
        n_samples = 1000
        x = np.random.uniform(low=-2., high=2., size=(n_samples, 1))
        y = 1.5 * np.sin(np.pi * x[:, 0]) + np.random.normal(loc=0., scale=1. * np.power(x[:, 0], 2))
        y = y.reshape([-1, 1]) / 5.
        x_train = x.reshape(-1)
        y_train = y.reshape(-1)
        if type_of_loss == 'qd':
            y_train = np.stack((y_train, y_train), axis=1)  # make this 2d so will be accepted
        x_test = np.random.uniform(low=-2., high=2., size=20)
        y_test = 1.5 * np.sin(np.pi * x_test) + np.random.normal(loc=0., scale=1. * np.power(x_test, 2))
        y_test = y_test / 5.
        x_border = np.linspace(-2, 2, 100)  # for evaluation plots
    elif type_of_data == '2':
        def f_predictable(xval):
            return xval + np.sin(np.pi * xval / 2)

        def f(xval, std=0.2):
            return f_predictable(xval) + np.random.randn(len(xval)) * std

        def get_data(num, start=0, end=4):
            xval = np.sort(np.random.rand(num) * (end - start) + start)
            yval = f(xval)
            return xval.reshape(-1, 1), yval

        x_train, y_train = get_data(num=20000)
        if type_of_loss == 'qd':
            y_train = np.stack((y_train, y_train), axis=1)  # make this 2d so will be accepted
        x_test, y_test = get_data(num=1000)
        x_border = np.linspace(0, 4, 100)  # for evaluation plots

    test = qd_objective(y_train, x_train)


if __name__ == '__main__':
    main()
