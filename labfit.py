# Author: Ben Champion <bwc3252@rit.edu>

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

def _sq_md(xdata, ydata, px, py, xerr, yerr):
    # Squared Mahalanobis distances between (xdata, ydata) and (px, py).
    # This is essentially a way of measuring distances between points that
    # accounts for the relative uncertainties.
    return ((xdata - px) / xerr)**2 + ((ydata - py) / yerr)**2

def _residuals(f, xdata, ydata, xerr, yerr, params):
    # First, we need to find the values of x that, given our current params,
    # minimize the total squared Mahalanobis distances.
    res = opt.minimize(lambda x0 : np.sum(_sq_md(xdata, ydata, x0, f(x0, *params), xerr, yerr)),
            xdata, method="Powell")
    # With those x values, return those squared Mahalanobis distances as the
    # "residuals" needed by scipy.optimize.leastsq
    return _sq_md(xdata, ydata, res.x, f(res.x, *params), xerr, yerr)

def fit(f, xdata, ydata, xerr=None, yerr=None, p0=None):
    """
    Fits a (generally) nonlinear function f(x, *params) to data (xdata, ydata),
    accounting for uncertainties in both the independent and dependent
    variables.

    Parameters
    ----------
    f:
        Function of the form f(x, param_1, param_2, ..., param_M), where x is
        an array containing values of the independent variable and each param_i
        is a float.
    xdata:
        Array of size N containing independent variable.
    ydata:
        Array of size N containing dependent variable.
    xerr:
        Uncertainty in the independent variable, must be a float or an array of
        size N.
    yerr:
        Uncertainty in the dependent variable, must be a float or an array of
        size N.
    p0:
        Array of size M containing initial guesses for parameter values.

    Returns
    -------
    params:
        Array of size M containing parameter estimates.

    cov:
        Array with shape (M, M) with parameter covariances.
    """
    # refine the initial guess by fitting with only uncertainty in the
    # dependent variable
    p0, cov0 = opt.curve_fit(f, xdata, ydata, sigma=yerr, p0=p0)
    # if there is no uncertainty in the independent variable, go ahead and
    # return the result of this fit
    if xerr is None:
        return p0, cov0
    # minimize the sum of the squared residuals wrt. params
    res = opt.leastsq(lambda p : _residuals(f, xdata, ydata, xerr, yerr, p),
            p0, full_output=True)
    params, cov, infodict, errmsg, ier = res
    # The remainder of this is from:
    # https://github.com/scipy/scipy/blob/v1.4.1/scipy/optimize/minpack.py, in
    # the leastsq function. Note that "cost", however, is calculated according
    # to our modified cost function, not the standard least squares cost.
    cost = np.sum(_residuals(f, xdata, ydata, xerr, yerr, params))
    s_sq = cost / (ydata.size - p0.size)
    cov *= s_sq
    return params, cov

### test code, does not run when imported

if __name__ == "__main__":
    # imports needed for test
    import matplotlib.pyplot as plt

    # random seed (for consistency between tests)
    np.random.seed(1)

    def g(x, a, b):
        return a * x + b

    #
    # TEST PARAMETERS
    #
    n = 20          # number of data points
    a = 1.0
    b = 0.0
    sigma_x = 0.2   # uncertainty in x
    sigma_y = 0.2   # uncertainty in y
    llim = 0.0
    rlim = 5.0
    xplot = np.linspace(llim, rlim, 100)
    
    print("true parameters:")
    print("\ta = {0:.3f}".format(a))
    print("\tb = {0:.3f}".format(b))
    
    # generate some data
    # make the actual errors a bit smaller than the uncertainties
    x = np.linspace(llim, rlim, n)
    y = g(x, a, b) + np.random.normal(0.0, sigma_y, n)
    x += np.random.normal(0.0, sigma_x, n)
    xerr = np.ones(n) * sigma_x
    yerr = np.ones(n) * sigma_y

    # close-ish initial guess (used by both fitting implementations)
    p0=np.array([1.4, -0.2])

    # fit using both x and y uncertainty
    params, cov = fit(g, x, y, xerr=xerr, yerr=yerr, p0=p0)
    std_dev = np.sqrt(np.diag(cov)) # uncertainty in the fit parameters

    # plot and print the results of this fit
    plt.plot(xplot, g(xplot, *params), label="Using $\sigma_x$ and $\sigma_y$")
    print("our results:")
    print("\ta = {0:.3f} +/- {1:.3f}".format(params[0], std_dev[0]))
    print("\tb = {0:.3f} +/- {1:.3f}".format(params[1], std_dev[1]))

    # fit using standard scipy curve_fit
    params, cov = opt.curve_fit(g, x, y, sigma=yerr, p0=p0)
    std_dev = np.sqrt(np.diag(cov)) # uncertainty in the fit parameters
    
    # plot and print the results of this fit
    plt.plot(xplot, g(xplot, *params), label="Using only $\sigma_y$")
    print("\ncurve_fit's results:")
    print("\ta = {0:.3f} +/- {1:.3f}".format(params[0], std_dev[0]))
    print("\tb = {0:.3f} +/- {1:.3f}".format(params[1], std_dev[1]))
    
    res = opt.minimize(lambda x0 : np.sum(_sq_md(x, y, x0, g(x0, *params), xerr, yerr)),
            x, method="Powell")
    plt.scatter(res.x, g(res.x, *params), label="Points used to calculate $M_D$"
        + " in last iteration")
    
    # plot the data points, generate the legend, and display the plot
    plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="none", color="black", capsize=2)
    plt.legend()
    plt.show()
