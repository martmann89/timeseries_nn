# Skewed-t Distribution by Hansen

class SkewStudent(Distribution, metaclass=AbstractDocStringInheritor):
    r"""
    Standardized Skewed Student's distribution for use with ARCH models

    Notes
    -----
    The Standardized Skewed Student's distribution ([1]_) takes two parameters,
    :math:`\eta` and :math:`\lambda`. :math:`\eta` controls the tail shape
    and is similar to the shape parameter in a Standardized Student's t.
    :math:`\lambda` controls the skewness. When :math:`\lambda=0` the
    distribution is identical to a standardized Student's t.

    References
    ----------
    .. [1] Hansen, B. E. (1994). Autoregressive conditional density estimation.
       *International Economic Review*, 35(3), 705–730.
       <https://www.ssc.wisc.edu/~bhansen/papers/ier_94.pdf>

    """

    def __init__(self, random_state: Optional[RandomState] = None) -> None:
        super().__init__(random_state=random_state)
        self._name = "Standardized Skew Student's t"
        self.num_params: int = 2

    def constraints(self) -> Tuple[NDArray, NDArray]:
        return array([[1, 0], [-1, 0], [0, 1], [0, -1]]), array([2.05, -300.0, -1, -1])

    def bounds(self, resids: NDArray) -> List[Tuple[float, float]]:
        return [(2.05, 300.0), (-1, 1)]

    def loglikelihood(
        self,
        parameters: Union[Sequence[float], ArrayLike1D],
        resids: ArrayLike,
        sigma2: ArrayLike,
        individual: bool = False,
    ) -> NDArray:
        r"""
        Computes the log-likelihood of assuming residuals are have a
        standardized (to have unit variance) Skew Student's t distribution,
        conditional on the variance.

        Parameters
        ----------
        parameters : ndarray
            Shape parameter of the skew-t distribution
        resids  : ndarray
            The residuals to use in the log-likelihood calculation
        sigma2 : ndarray
            Conditional variances of resids
        individual : bool, optional
            Flag indicating whether to return the vector of individual log
            likelihoods (True) or the sum (False)

        Returns
        -------
        ll : float
            The log-likelihood

        Notes
        -----
        The log-likelihood of a single data point x is

        .. math::

            \ln\left[\frac{bc}{\sigma}\left(1+\frac{1}{\eta-2}
                \left(\frac{a+bx/\sigma}
                {1+sgn(x/\sigma+a/b)\lambda}\right)^{2}\right)
                ^{-\left(\eta+1\right)/2}\right],

        where :math:`2<\eta<\infty`, and :math:`-1<\lambda<1`.
        The constants :math:`a`, :math:`b`, and :math:`c` are given by

        .. math::

            a=4\lambda c\frac{\eta-2}{\eta-1},
                \quad b^{2}=1+3\lambda^{2}-a^{2},
                \quad c=\frac{\Gamma\left(\frac{\eta+1}{2}\right)}
                {\sqrt{\pi\left(\eta-2\right)}
                \Gamma\left(\frac{\eta}{2}\right)},

        and :math:`\Gamma` is the gamma function.
        """
        eta, lam = parameters

        const_c = self.__const_c(parameters)
        const_a = self.__const_a(parameters)
        const_b = self.__const_b(parameters)

        resids = resids / sigma2 ** 0.5
        lls = log(const_b) + const_c - log(sigma2) / 2
        if abs(lam) >= 1.0:
            lam = sign(lam) * (1.0 - 1e-6)
        llf_resid = (
            (const_b * resids + const_a) / (1 + sign(resids + const_a / const_b) * lam)
        ) ** 2
        lls -= (eta + 1) / 2 * log(1 + llf_resid / (eta - 2))

        if individual:
            return lls
        else:
            return sum(lls)

    def starting_values(self, std_resid: ArrayLike1D) -> NDArray:
        """
        Construct starting values for use in optimization.

        Parameters
        ----------
        std_resid : ndarray
            Estimated standardized residuals to use in computing starting
            values for the shape parameter

        Returns
        -------
        sv : ndarray
            Array containing starting valuer for shape parameter

        Notes
        -----
        Uses relationship between kurtosis and degree of freedom parameter to
        produce a moment-based estimator for the starting values.
        """
        k = stats.kurtosis(std_resid, fisher=False)
        sv = max((4.0 * k - 6.0) / (k - 3.0) if k > 3.75 else 12.0, 4.0)
        return array([sv, 0.0])

    def _simulator(self, size: Union[int, Tuple[int, ...]]) -> NDArray:
        # No need to normalize since it is already done in parameterization
        assert self._parameters is not None
        ppf = self.ppf(self._random_state.random_sample(size=size), self._parameters)
        assert isinstance(ppf, ndarray)
        return ppf

    def simulate(
        self, parameters: Union[int, float, Sequence[Union[float, int]], ArrayLike1D]
    ) -> Callable[[Union[int, Tuple[int, ...]]], NDArray]:
        parameters = ensure1d(parameters, "parameters", False)
        if parameters[0] <= 2.0:
            raise ValueError("The shape parameter must be larger than 2")
        if abs(parameters[1]) > 1.0:
            raise ValueError(
                "The skew parameter must be smaller than 1 in absolute value"
            )
        self._parameters = parameters
        return self._simulator

    def parameter_names(self) -> List[str]:
        return ["nu", "lambda"]

    def __const_a(self, parameters: Union[NDArray, Sequence[float]]) -> float:
        """
        Compute a constant.

        Parameters
        ----------
        parameters : ndarray
            Shape parameters of the skew-t distribution

        Returns
        -------
        a : float
            Constant used in the distribution

        """
        eta, lam = parameters
        c = self.__const_c(parameters)
        return float(4 * lam * exp(c) * (eta - 2) / (eta - 1))

    def __const_b(self, parameters: Union[NDArray, Sequence[float]]) -> float:
        """
        Compute b constant.

        Parameters
        ----------
        parameters : ndarray
            Shape parameters of the skew-t distribution

        Returns
        -------
        b : float
            Constant used in the distribution
        """
        lam = float(parameters[1])
        a = self.__const_a(parameters)
        return (1 + 3 * lam ** 2 - a ** 2) ** 0.5

    @staticmethod
    def __const_c(parameters: Union[NDArray, Sequence[float]]) -> float:
        """
        Compute c constant.

        Parameters
        ----------
        parameters : ndarray
            Shape parameters of the skew-t distribution

        Returns
        -------
        c : float
            Log of the constant used in loglikelihood
        """
        eta = parameters[0]
        # return gamma((eta+1)/2) / ((pi*(eta-2))**.5 * gamma(eta/2))
        return float(gammaln((eta + 1) / 2) - gammaln(eta / 2) - log(pi * (eta - 2)) / 2)

    def cdf(
        self,
        resids: ArrayLike,
        parameters: Optional[Union[Sequence[float], ArrayLike1D]] = None,
    ) -> NDArray:
        parameters = self._check_constraints(parameters)
        scalar = isscalar(resids)
        if scalar:
            resids = array([resids])

        eta, lam = parameters

        a = self.__const_a(parameters)
        b = self.__const_b(parameters)

        var = eta / (eta - 2)
        y1 = (b * resids + a) / (1 - lam) * sqrt(var)
        y2 = (b * resids + a) / (1 + lam) * sqrt(var)
        tcdf = stats.t(eta).cdf
        resids = asarray(resids)
        p = (1 - lam) * tcdf(y1) * (resids < (-a / b))
        p += (resids >= (-a / b)) * ((1 - lam) / 2 + (1 + lam) * (tcdf(y2) - 0.5))
        if scalar:
            p = p[0]
        return p

    def ppf(
        self,
        pits: Union[Sequence[float], ArrayLike1D],
        parameters: Optional[Union[Sequence[float], ArrayLike1D]] = None,
    ) -> Union[float, NDArray]:
        parameters = self._check_constraints(parameters)
        scalar = isscalar(pits)
        if scalar:
            pits = array([pits])
        pits = asarray(pits)
        eta, lam = parameters

        a = self.__const_a(parameters)
        b = self.__const_b(parameters)

        cond = pits < (1 - lam) / 2

        icdf1 = stats.t.ppf(pits[cond] / (1 - lam), eta)
        icdf2 = stats.t.ppf(0.5 + (pits[~cond] - (1 - lam) / 2) / (1 + lam), eta)
        icdf = -999.99 * ones_like(pits)
        assert isinstance(icdf, ndarray)
        icdf[cond] = icdf1
        icdf[~cond] = icdf2
        icdf = icdf * (1 + sign(pits - (1 - lam) / 2) * lam) * (1 - 2 / eta) ** 0.5 - a
        icdf = icdf / b

        if scalar:
            return float(icdf[0])
        assert isinstance(icdf, ndarray)
        return icdf

    def moment(
        self, n: int, parameters: Optional[Union[Sequence[float], ArrayLike1D]] = None
    ) -> float:
        parameters = self._check_constraints(parameters)
        eta, lam = parameters

        if n < 0 or n >= eta:
            return nan

        a = self.__const_a(parameters)
        b = self.__const_b(parameters)

        loc = -a / b
        lscale = sqrt(1 - 2 / eta) * (1 - lam) / b
        rscale = sqrt(1 - 2 / eta) * (1 + lam) / b

        moment = 0.0
        for k in range(n + 1):  # binomial expansion around loc
            # 0->inf right partial moment for ordinary t(eta)
            r_pmom = (
                0.5
                * (gamma(0.5 * (k + 1)) * gamma(0.5 * (eta - k)) * eta ** (0.5 * k))
                / (sqrt(pi) * gamma(0.5 * eta))
            )
            l_pmom = ((-1) ** k) * r_pmom

            lhs = (1 - lam) * (lscale ** k) * (loc ** (n - k)) * l_pmom
            rhs = (1 + lam) * (rscale ** k) * (loc ** (n - k)) * r_pmom
            moment += comb(n, k) * (lhs + rhs)

        return moment

    def partial_moment(
        self,
        n: int,
        z: float = 0.0,
        parameters: Optional[Union[Sequence[float], ArrayLike1D]] = None,
    ) -> float:
        parameters = self._check_constraints(parameters)
        eta, lam = parameters

        if n < 0 or n >= eta:
            return nan

        a = self.__const_a(parameters)
        b = self.__const_b(parameters)

        loc = -a / b
        lscale = sqrt(1 - 2 / eta) * (1 - lam) / b
        rscale = sqrt(1 - 2 / eta) * (1 + lam) / b

        moment = 0.0
        for k in range(n + 1):  # binomial expansion around loc
            lbound = min(z, loc)
            lhs = (
                (1 - lam)
                * (loc ** (n - k))
                * (lscale ** k)
                * StudentsT._ord_t_partial_moment(k, z=(lbound - loc) / lscale, nu=eta)
            )

            if z > loc:
                rhs = (
                    (1 + lam)
                    * (loc ** (n - k))
                    * (rscale ** k)
                    * (
                        StudentsT._ord_t_partial_moment(k, z=(z - loc) / rscale, nu=eta)
                        - StudentsT._ord_t_partial_moment(k, z=0.0, nu=eta)
                    )
                )
            else:
                rhs = 0.0

            moment += comb(n, k) * (lhs + rhs)

        return moment
