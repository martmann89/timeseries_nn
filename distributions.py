import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, pareto, t, rv_continuous, beta
from scipy.special import gamma
# from sympy.stats import ContinuousRV, Var


def heaviside(x):
    return (1+np.sign(x))/2


class MyUnitStudentPdf(rv_continuous):
    def _pdf(self, x, df):
        return gamma((df+1)/2)/(np.sqrt(np.pi*(df-2))*gamma(df/2))*(1+(x**2/(df-2)))**(-(df+1)/2)


class MytPdf(rv_continuous):
    def _pdf(self, x, df):
        return gamma((df+1)/2)/(np.sqrt(np.pi*df)*gamma(df/2))*(1+(x**2/df))**(-(df+1)/2)


class SkewedStudentT(rv_continuous):
    def _pdf(self, x, df, lam):
        c = gamma((df+1)/2)/(np.sqrt(np.pi*(df-2))*gamma(df/2))
        a = 4*lam*c*((df-2)/(df-1))
        b = np.sqrt(1 + 3*lam**2 - a**2)
        if isinstance(x, float):
            if x < -a/b:
                d = 1-lam
            else:
                d = 1+lam
        else:
            d = np.zeros(len(x))
            # d[x < -a/b] = 1-lam[0]
            d[x < -a/b] = 1-lam
            # d[x >= -a/b] = 1+lam[0]
            d[x >= -a/b] = 1+lam
        return b*c*(1+(1/(df-2))*((b*x+a)/d)**2)**(-(df+1)/2)

    def _argcheck(self, df, lam):
        df_bool = 2 < df < 30
        lam_bool = -1 < lam < 1
        return df_bool & lam_bool


class TimeVaryingSkewedStudentT(rv_continuous):
    def _pdf(self, x, df1, df2, df3, lam1, lam2, lam3):
        lower_df, upper_df = 2.1, 30
        lower_lam, upper_lam = -0.95, 0.95
        df = df1 + df2*x[-1] + df3*x[-1]**2
        df = lower_df + (upper_df-lower_df)/(1+np.exp(-df))
        lam = lam1 + lam2*x[-1] + lam3*x[-1]**2
        lam = lower_lam + (upper_lam-lower_lam)/(1+np.exp(-lam))
        c = gamma((df+1)/2)/(np.sqrt(np.pi*(df-2))*gamma(df/2))
        a = 4*lam*c*((df-2)/(df-1))
        b = np.sqrt(1 + 3*lam**2 - a**2)
        d = np.zeros(len(x))
        d[x < -a/b] = 1-lam
        d[x >= -a/b] = 1+lam
        return b*c*(1+(1/(df-2))*((b*x+a)/d)**2)**(-(df+1)/2)


class SkewedNorm(rv_continuous):
    def _pdf(self, x, lam):
        return 2/(lam+(1/lam))*(norm.pdf(lam*x)*heaviside(-x)+norm.pdf((x/lam))*heaviside(x))


def main():
    # my_unit_t = MyUnitStudentPdf(a=-100, b=100, name='my_skewed_pdf')
    # my_t_distr = MytPdf(a=-8, b=8, name='my_t_pdf')
    skewed_stu_t = SkewedStudentT(a=-10, b=10, shapes='df, lam', name='Skewed Student T (Hansen)')
    tv_skewed_stu_t = TimeVaryingSkewedStudentT(a=-10, b=10, name='Time Varying Skewed Student T (Hansen)')
    # skewed_norm = SkewedNorm(a=-5, b=5, name='Skewed Normal Distribution')
    # lam = 1.5
    df = 7
    x = np.linspace(-3, 3, 100)
    lams = [-0.5]
    # a, b = 1., 2.
    # x_rv = beta.rvs(a, b, size=1000)
    # a1, b1, loc1, scale1 = beta.fit(x_rv)
    # print(a1, b1, loc1, scale1)
    for lam in lams:
        # y = skewed_norm(lam)
        # z = my_t_distr(df)
        z = skewed_stu_t(df, lam)
        # plt.plot(x, z.pdf(x))
        z_rv = z.rvs(1000)
        # est_par = skewed_stu_t.fit(z_rv, 2.5, 0.1, floc=0, fscale=1)
        est_par = tv_skewed_stu_t.fit(z_rv, 0,0,0,0,0,0, loc=0, scale=1, floc=0, fscale=1)
        print(est_par)
        # print(np.mean(z_rv), np.var(z_rv))
        # me = z.mean()
        # print(y.var())


def plot_pdf(distr, par):
    x = np.linspace(distr.ppf(0.01, par), distr.ppf(0.99, par), 100)
    print(distr.stats(par, moments='mv'))
    plt.plot(x, distr.pdf(x, par), '-')


def skewed_dist(x, lam):
    return 2/(lam+(1/lam))*(norm.pdf(lam*x)*heaviside(-x)+norm.pdf((x/lam))*heaviside(x))


def my_t(x, df):
    return gamma((df+1)/2)/(np.sqrt(np.pi*(df-2))*gamma(df/2))*(1+(x**2/(df-2)))**(-(df+1)/2)


if __name__ == '__main__':
    main()
