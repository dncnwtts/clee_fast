'''
I'm trying to regress a multidimensional polynomial out of the theoretical power
spectra C_l(r,s,tau). I believe this is all done in scikit-learn (a little bit
of googling got me there) and I have done this with a simple example.
'''

import numpy as np
import matplotlib.pyplot as plt


param_file = '../data/training_params_tau.txt'
EE_file = '../data/training_data_tau_EE.txt'
BB_file = '../data/training_data_tau_BB.txt'


import camb
pars = camb.CAMBparams()
pars.set_cosmology(H0=70., ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
pars.InitPower.set_params(ns=1.0, r=0.0, As=2.3e-9)
pars.set_for_lmax(200, lens_potential_accuracy=0)
pars.WantTensors = True
pars.AccurateReionization = 1

def C_l(tau, ps='EE'):
    pars.set_cosmology(H0=70, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=tau)
    pars.InitPower.set_params(ns=1.0, r=0, As=2.3e-9)
    pars.set_for_lmax(2500, lens_potential_accuracy=0)
    pars.WantTensors = True
    pars.AccurateReionization = 1
    
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars)
    cl = powers['total']*(1e6*pars.TCMB)**2 # in uK_CMB^2
    if ps == 'EE':
        return cl[:200,1]
    elif ps == 'BB':
        return cl[:200,2]
    elif ps == 'EB':
        return cl[:200,1], cl[:200,2]
    else:
        return cl


def example():
    EE = C_l(0.07)
    ell = np.arange(EE.size)
    plt.figure()
    plt.plot(ell, EE)
    plt.xlim([2, ell.max()])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\ell$', size=20)
    plt.ylabel(r'$D_\ell^\mathrm{EE}$', size=20)
    plt.show()


consider = 'EE'
degree = 9
new = True
show = True
ntrain = 2000
widths = np.array([0.08])
centers = np.array([0.06])
if new == True:
    try:
        points = np.loadtxt(param_file)
        values_EE = np.loadtxt(EE_file)
        values_BB = np.loadtxt(BB_file)
        points = points.reshape(-1,1)
    except IOError:
        points = np.array([]).reshape(-1,1)
        values_EE = np.array([]).reshape(-1, 200)
        values_BB = np.array([]).reshape(-1, 200)
    # Only needs to be random when we are cursed by dimensionality.
    new = np.linspace(centers-widths/2, centers+widths/2, ntrain).reshape(-1,1)
    points = np.concatenate( (points, new))
    for i in range(ntrain):
        EE, BB = C_l(*new[i], ps='EB')
        values_EE = np.concatenate((values_EE, EE.reshape(-1,200)))
        values_BB = np.concatenate((values_BB, BB.reshape(-1,200)))
        if i % 10 == 0:
            print(i, new[i])
    print('\n\n\n\n')
    np.savetxt(EE_file, values_EE)
    np.savetxt(BB_file, values_BB)
    np.savetxt(param_file, points)
else:
    values_EE = np.loadtxt(EE_file)
    values_BB = np.loadtxt(BB_file)
    points = np.loadtxt(param_file)
if consider == 'EE':
    values = values_EE
else:
    values = values_BB

ntest = len(values)/10
np.random.seed(0) # For reproducibility.
predict = widths*(np.random.rand(ntest,1) - 0.5) + centers

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly = PolynomialFeatures(degree=degree)
points = points.reshape(-1,1)
X_ = poly.fit_transform(points) # is this just vandermonde?

import time
t0 = time.time()

predict_ = poly.fit_transform(predict)

clf = LinearRegression()
estimate = []
for l in range(values.shape[1]):
    values_l = values[:,l]
    clf.fit(X_, values_l)
    estimate_l = clf.predict(predict_)
    estimate.append(estimate_l)
estimate = np.array(estimate)
t1 = time.time()
print('Takes {0} seconds to evaluate {1} spectra with {2} training values.\n'.format(t1-t0, len(predict), len(values)))

for i, p in enumerate(predict[:50]):
    plt.figure('percent difference')
    true = C_l(*p, ps=consider)
    esti = estimate[:,i]
    x = (true[2:]-esti[2:])/true[2:]
    plt.plot(x, label=r'$\tau={0}$'.format(*p))

if show:
    
    plt.figure('percent difference')
    plt.xscale('log')
    plt.xlim([2,200])
    plt.ylim([-1e-2, 1e-2])
    plt.ylabel(r'$\Delta C_\ell/C_\ell$', size=20)
    plt.xlabel(r'$\ell$', size=20)
    plt.savefig('estimate_accuracy_tau')
    
    
    plt.figure()
    plt.hist(points[:,0], bins=np.linspace(centers-widths/2, centers+widths/2, 100), histtype='stepfilled')
    plt.xlabel(r'$\tau$', size=20)
    plt.savefig('../plots/tau_density')
    
    plt.show()
