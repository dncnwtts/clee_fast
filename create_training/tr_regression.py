'''
I'm trying to regress a multidimensional polynomial out of the theoretical power
spectra C_l(r,s,tau). I believe this is all done in scikit-learn (a little bit
of googling got me there) and I have done this with a simple example.

Using PlanckTT+lowP+lensing best fit, the "reference cosmology".
\omega_b=0.02226
\omega_c=0.1197
\Omega_m=0.308
n_s=0.9677
H_0=67.81
Y_P=0.2453
tau=0.058
ln10^10A_s=3.062
A_s=2.13e-9
'''

import numpy as np
import matplotlib.pyplot as plt

import camb
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.81, ombh2=0.02226, omch2=0.1197, mnu=0.06, omk=0, tau=0.058)
pars.InitPower.set_params(ns=0.9677, r=0.1, As=2.13e-9)
pars.set_for_lmax(200, lens_potential_accuracy=0)
pars.WantTensors = True
pars.AccurateReionization = 1

def C_l(r, tau, ps='BB'):
    pars.set_cosmology(H0=67.81, ombh2=0.02226, omch2=0.1197, mnu=0.06, omk=0, tau=tau)
    pars.InitPower.set_params(ns=0.9677, r=r, As=2.13e-9)
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
    BB = C_l(0.01, 0.07)
    ell = np.arange(BB.size)
    plt.figure()
    plt.plot(ell, BB)
    plt.xlim([2, ell.max()])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\ell$', size=20)
    plt.ylabel(r'$D_\ell^\mathrm{BB}$', size=20)
    plt.show()



consider = 'EE'
degree = 7
new = False
show = True
#ntrain = 5
ntrain = 5000
widths = np.array([0.10, 0.16])
centers = np.array([0.05, 0.12])
if new == True:
    try:
        points = np.loadtxt('training_params_rt.txt')
        values_EE = np.loadtxt('training_data_EE_rt.txt')
        values_BB = np.loadtxt('training_data_BB_rt.txt')
    except IOError:
        points = np.array([]).reshape(-1,2)
        values_EE = np.array([]).reshape(-1, 200)
        values_BB = np.array([]).reshape(-1, 200)
    new = widths*(np.random.rand(ntrain,2) - 0.5) + centers
    points = np.concatenate( (points, new))
    for i in range(ntrain):
        EE, BB = C_l(*new[i], ps='EB')
        values_EE = np.concatenate((values_EE, EE.reshape(-1,200)))
        values_BB = np.concatenate((values_BB, BB.reshape(-1,200)))
        if i % 10 == 0:
            print(i, new[i])
    print('\n\n\n\n')
    np.savetxt('../data/training_data_EE_rt.txt', values_EE)
    np.savetxt('../data/training_data_BB_rt.txt', values_BB)
    np.savetxt('../data/training_params_rt.txt', points)
else:
    values_EE = np.loadtxt('../data/training_data_EE_rt.txt')
    values_BB = np.loadtxt('../data/training_data_BB_rt.txt')
    points = np.loadtxt('../data/training_params_rt.txt')
if consider == 'EE':
    values = values_EE
else:
    values = values_BB

ntest = len(values)/10
predict = widths*(np.random.rand(ntest,2) - 0.5) + centers

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly = PolynomialFeatures(degree=degree)
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
    plt.plot(x, label=r'$r={0}, \tau={1}$'.format(*p))

if show:
    
    plt.figure('percent difference')
    plt.xscale('log')
    plt.xlim([2,200])
    plt.ylim([-1e-2, 1e-2])
    plt.ylabel(r'$\Delta C_\ell/C_\ell$', size=20)
    plt.xlabel(r'$\ell$', size=20)
    plt.savefig('../plots/estimate_accuracy_ts')
    
    
    import corner
    corner.corner(points, labels=[r'$r$', r'$\tau$'])
    plt.savefig('../plots/point_density_rt')
    
    
    ell = 9
    values = values[:,ell+2]
    clf.fit(X_, values)
    estimate = clf.predict(predict_)
    
    mini, maxi = values.min(), values.max()

    plt.figure() 
    plt.scatter(points[:,0], points[:,1], c=values, vmin=mini, vmax=maxi)
    plt.scatter(predict[:,0], predict[:,1], c=estimate, s=100, edgecolors='white', vmin=mini, vmax=maxi)
    plt.colorbar()
    plt.xlabel(r'$r$', size=20)
    plt.ylabel(r'$\tau$', size=20)
    plt.title(r'$C_{{\ell=10}}^\mathrm{{ {0} }}(r,\tau)$ (fit with degree-{1} polynomial)'.format(consider, degree), size=20)

    plt.savefig('../plots/single_ell_accuracy_rt.png')
    plt.show()
