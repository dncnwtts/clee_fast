'''
This assumes that we have already gotten the E and B spectra computed from CAMB.
This should be very fast.
'''

# degree of the multivariate polynomial to be fit
degree = 9
# determines whether or not to use E- or B-mode power spectrum.
consider = 'EE' #'BB'

import numpy as np
import matplotlib.pyplot as plt

# PolynomialFeatures essentially creates a Vandermonde matrix in the parameters
# you're interested in, making all possible variable combinations up to order
# degree, i.e. x^5, x^4..., x^4y, x^3y,... 1.
from sklearn.preprocessing import PolynomialFeatures
# LinearRegression creates the pipeline that fits the model that we're
# interested in. Here, the model is a linear one, where the coefficients of the
# polynomial are the parameters that are fit.
from sklearn.linear_model import LinearRegression
# Training data.
global values_EE, values_BB, points
# These are pre-computed by the trs_regression.py file. The more points
# computed, the more precisely the regression fits the true solution, but the
# time goes up as O(C^2*N), where N is the number of training samples and C is
# the degree of the polynomial.
values_EE = np.loadtxt('data/training_data_tau_EE.txt')
values_BB = np.loadtxt('data/training_data_tau_BB.txt')
points = np.loadtxt('data/training_params_tau.txt')

def get_cl(tau, consider='EE', degree=5):
    if consider == 'EE':
        values = values_EE
    else:
        values = values_BB

    v = values#[:100]
    p = points#[:100]

    poly = PolynomialFeatures(degree=degree)
    # Vandermonde matrix of pre-computed paramter values.
    X_ = poly.fit_transform(p.reshape(-1,1))

    predict = np.array([tau]).reshape(1,-1)
    # Creates matrix of values you want to estimate from the existing
    # measurements. Computation speed scales very slowly when you ask for
    # estimate of many sets of parameters.
    predict_ = poly.fit_transform(predict)

    clf = LinearRegression()
    estimate = []
    for l in range(2, v.shape[1]):
        values_l = v[:,l]
        clf.fit(X_, values_l)
        estimate_l = clf.predict(predict_)
        estimate.append(estimate_l)
    estimate = np.array(estimate)

    ell = np.arange(2, l+1)
    Z = 2*np.pi/(ell*(ell+1))
    return ell, Z*estimate[:,0]


if __name__ == '__main__':
    # Sample computation.
    color_idx = np.linspace(0,1, 10)
    taus = np.linspace(0.03, 0.1, 10)
    times = []
    import time
    for ind, tau in zip(color_idx, taus):
        t0 = time.time()
        ell, Cl = get_cl(tau, consider=consider)
        times.append(time.time()-t0)
        plt.loglog(ell, Cl, color=plt.cm.viridis(ind), alpha=0.8, lw=5)
    plt.xlim([2, 200])
    plt.xlabel(r'$\ell$', size=20)
    plt.ylabel(r'$C_\ell^\mathrm{{ {0} }}$'.format(consider), size=20)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
            norm=plt.Normalize(vmin=taus.min(), vmax=taus.max()))
    sm._A = []
    plt.colorbar(sm, label=r'$\tau$')
    plt.savefig('plots/tau_example.png')
    plt.show()
    
    print('Takes ~{0} seconds'.format(round(np.mean(times),2)))
