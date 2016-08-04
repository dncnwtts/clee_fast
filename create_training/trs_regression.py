'''
I'm trying to regress a multidimensional polynomial out of the theoretical power
spectra C_l(r,s,tau). I believe this is all done in scikit-learn (a little bit
of googling got me there) and I have done this with a simple example.
'''

import numpy as np
import matplotlib.pyplot as plt

import camb
pars = camb.CAMBparams()
pars.set_cosmology(H0=70., ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
pars.InitPower.set_params(ns=1.0, r=0.1, As=2.3e-9)
pars.set_for_lmax(200, lens_potential_accuracy=0)
pars.WantTensors = True
pars.AccurateReionization = 1

def C_l(r, s, tau, ps='BB'):
    pars.set_cosmology(H0=70, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=tau)
    pars.InitPower.set_params(ns=1.0, r=r, As=2.3e-9*s)
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
    BB = C_l(0, 1, 0.07)
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
ntrain = 5000
widths = np.array([0.1, 0.5, 0.06])
centers = np.array([0.05, 1.0, 0.06])
if new == True:
    try:
        points = np.loadtxt('training_params.txt')
        values_EE = np.loadtxt('training_data_EE.txt')
        values_BB = np.loadtxt('training_data_BB.txt')
    except IOError:
        points = np.array([]).reshape(-1,3)
        values_EE = np.array([]).reshape(-1, 200)
        values_BB = np.array([]).reshape(-1, 200)
    new = widths*(np.random.rand(ntrain,3) - 0.5) + centers
    points = np.concatenate( (points, new))
    for i in range(ntrain):
        EE, BB = C_l(*new[i], ps='EB')
        values_EE = np.concatenate((values_EE, EE.reshape(-1,200)))
        values_BB = np.concatenate((values_BB, BB.reshape(-1,200)))
        if i % 10 == 0:
            print(i, new[i])
    print('\n\n\n\n')
    np.savetxt('../data/training_data_EE.txt', values_EE)
    np.savetxt('../data/training_data_BB.txt', values_BB)
    np.savetxt('../data/training_params.txt', points)
else:
    values_EE = np.loadtxt('../data/training_data_EE.txt')
    values_BB = np.loadtxt('../data/training_data_BB.txt')
    points = np.loadtxt('../data/training_params.txt')
if consider == 'EE':
    values = values_EE
else:
    values = values_BB

ntest = len(values)/10
predict = widths*(np.random.rand(ntest,3) - 0.5) + centers

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
    plt.plot(x, label=r'$r={0}, s={1}, \tau={2}$'.format(*p))

if show:
    
    plt.figure('percent difference')
    plt.xscale('log')
    plt.xlim([2,200])
    plt.ylim([-1e-2, 1e-2])
    plt.ylabel(r'$\Delta C_\ell/C_\ell$', size=20)
    plt.xlabel(r'$\ell$', size=20)
    plt.savefig('../plots/estimate_accuracy_trs')
    
    
    import corner
    corner.corner(points, labels=[r'$r$', r'$s$', r'$\tau$'])
    plt.savefig('../plots/point_density_trs')
    
    
    ell = 10 
    values = values[:,ell+2]
    clf.fit(X_, values)
    estimate = clf.predict(predict_)
    
    mini, maxi = values.min(), values.max()
    
    fig, axes = plt.subplots(2, 2, sharex='col', sharey='row')
    bigax = fig.add_subplot(111)
    fig.delaxes(axes[0,1])
    bottomleft = plt.subplot(223)
    bottomleft.scatter(points[:,0], points[:,2], c=values, vmin=mini,
            vmax=maxi)
    plt.scatter(predict[:,0], predict[:,2], c=estimate, s=100,
            edgecolors='white', vmin=mini, vmax=maxi)
    plt.xlabel(r'$r$', size=20)
    plt.ylabel(r'$\tau$', size=20)
    plt.xlim([0, 0.11])
    plt.ylim([0.02, 0.095])
    
    upperleft = plt.subplot(221)
    plt.scatter(points[:,0], points[:,1], c=values, vmin=mini,
            vmax=maxi)
    plt.scatter(predict[:,0], predict[:,1], c=estimate, s=100,
            edgecolors='white', vmin=mini, vmax=maxi)
    plt.ylabel(r'$s$', size=20)
    plt.xlim([0, 0.11])
    
    bottomright = plt.subplot(224)
    plt.scatter(points[:,1], points[:,2], c=values, vmin=mini,
            vmax=maxi)
    plt.scatter(predict[:,1], predict[:,2], c=estimate, s=100,
            edgecolors='white', vmin=mini, vmax=maxi)
    plt.xlabel(r'$s$', size=20)
    plt.ylim([0.02, 0.095])
    plt.xlim([0.75, 1.3])
    cbaxes = fig.add_axes([0.9, 0.1, 0.03, 0.8])
    plt.colorbar(cax=cbaxes)


    plt.setp(upperleft.get_xticklabels(), visible=False)
    plt.setp(bottomright.get_yticklabels(), visible=False)
    plt.subplots_adjust(hspace=0, wspace=0)
    fig.suptitle(r'$C_{{ \ell={0} }}^{{ \mathrm{{ {1} }} }}$ (degree-{2} polynomial)'.format(ell, consider, degree), size=20)

    plt.savefig('../plots/single_ell_accuracy_trs.png')
    plt.show()
