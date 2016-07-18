import main_tau as clfast
import numpy as np
import matplotlib.pyplot as plt

tau = 0.07
ell, cl = clfast.get_cl(tau)

plt.loglog(ell, cl, 'k', lw=5)
plt.xlabel(r'$\ell$', size=20)
plt.ylabel(r'$C_\ell^\mathrm{EE}$', size=20)
plt.show()
