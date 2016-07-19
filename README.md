# clee-fast
Uses scikit-learn to quickly obtain polarized power spectra as function of tau, r, and A_s.

## Shell usage
```
>>> import main_ts
>>> tau, s = 0.065, 1.05
>>> ell, cl = main_ts.get_cl(tau, s)
```

## Example usage
```
python main_tau.py
```
The output of this is several evaluations of C_ell, each of which take about 0.1 seconds to run.
![alt text](https://github.com/pqrs6/clee-fast/blob/master/plots/tau_example.png "dummy text")

### Relative accuracy of fit
![alt text](https://github.com/pqrs6/clee-fast/blob/master/plots/estimate_accuracy.png "dummy text")



## Simplest example
```
python example.py
```
The output of this function is a single power spectrum as a function of tau.

![alt text](https://github.com/pqrs6/clee-fast/blob/master/plots/simplest.png "dummy text")
