import numpy as np
import matplotlib.pyplot as plt
from D_models.SkewStudent import SkewStudent


def plot_skew_functions():
    fig, ax = plt.subplots(2, 2, figsize=(8, 8))

    x = np.linspace(-3, 3, 100)
    x1 = np.linspace(0.01, 0.99, 100)
    eta = 3
    lams = [-0.5, 0, 0.2]

    for lam in lams:
        z = SkewStudent(eta=eta, lam=lam)
        ax[0, 0].plot(x, z.pdf(x), label='$\lambda$ = ' + str(lam))
    ax[0, 0].set_title('pdf')
    ax[0, 0].set(xlabel='z', ylabel='$\mathrm{skst}(z|\eta, \lambda)$')
    # ax[0, 0].ylabel('$skst(z|\eta, \lambda)$')
    ax[0, 0].legend()

    for lam in lams:
        z = SkewStudent(eta=eta, lam=lam)
        ax[0, 1].plot(x, z.cdf(x), label='$\lambda$ = ' + str(lam))
    ax[0, 1].set_title('cdf')
    ax[0, 1].set(xlabel='z', ylabel='$\mathrm{SKST}(z|\eta, \lambda)$')
    # ax[0, 1].ylabel('$SKST(z|\eta, \lambda)$')
    ax[0, 1].legend()

    for lam in lams:
        z = SkewStudent(eta=eta, lam=lam)
        ax[1, 0].plot(x1, z.ppf(x1), label='$\lambda$ = ' + str(lam))
    ax[1, 0].set_title('inverse cdf')
    ax[1, 0].set(xlabel='z', ylabel='$\mathrm{SKST}(z|\eta, \lambda)^{-1}$')
    # ax[1, 0].ylabel('$SKST(z|\eta, \lambda)$')
    ax[1, 0].legend()

    # ax[1, 1].set_position([0.55, 0.125, 0.228, 0.343])
    plt.tight_layout()

    ax[1, 1].set_visible(False)
    ax[1, 0].set_position([0.33, 0.08, 0.4, 0.4])
    plt.show()


if __name__ == '__main__':
    plot_skew_functions()
