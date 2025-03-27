import math
import numpy as np
import matplotlib.pyplot as plt

from mpmath import mp

from config import FIGURES_DIR

"""
    Purpose: Show the log1mexp stabilizes log(1-exp(-|x|)) for x > 0.

    Plot inspired by Figure 1 in "Accurately Computing log(1 − exp(− |a|)) Assessed by the Rmpfr package" by Martin Mächler.
"""

dtype = np.float32
x_values = np.logspace(-149, 1, 2000, base=2).astype(dtype)

# Compute the high precision values
mp.dps = 1024 # number of decimal places
exact_values = np.array([mp.log(1 - mp.exp(-a)) for a in x_values], dtype=np.float64)


## Two methods currently used in plot ##
def log1mexp_method(a):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = a < math.log(2) # x < 0
        return np.where(
            mask,
            np.log(-np.expm1(-a, dtype=dtype), dtype=dtype),
            np.log1p(-np.exp(-a, dtype=dtype), dtype=dtype),
        )
    
def default_method(a):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.log(1 - np.exp(-a, dtype=dtype), dtype=dtype)

## Following two not included, but are in the original plot ##
def expm1_method(a):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.log(-np.expm1(-a, dtype=dtype), dtype=dtype)

def log1p_method(a):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.log1p(-np.exp(-a, dtype=dtype), dtype=dtype)


def relative_error(computed, exact):
    return np.abs(1 - computed / exact)

# Compute the values using the three methods with limited precision
default_values = default_method(x_values)
log1mexp_values = log1mexp_method(x_values)
#expm1_values = expm1_method(x_values)
#log1p_values = log1p_method(x_values)

# Compute the relative errors
relative_error_default = relative_error(default_values, exact_values)
relative_error_log1mexp = relative_error(log1mexp_values, exact_values)
#relative_error_expm1 = relative_error(expm1_values, exact_values)
#relative_error_log1p = relative_error(log1p_values, exact_values)


plt.figure(figsize=(5, 2))
plt.plot(x_values, relative_error_default, label=r'$\mathtt{log}\left(1 - \mathtt{exp}\left(x\right)\right)$', linestyle='-', color='red', alpha=0.5)
#plt.plot(x_values, relative_error_log1mexp, label=r'\texttt{log1mexp}($-x$)', linestyle=':', color='blue', alpha=0.5)
plt.plot(x_values, relative_error_log1mexp, label=r'$\mathtt{log1mexp}(x)$', linestyle=':', color='blue', alpha=0.5)

#plt.plot(x_values, relative_error_expm1, label='log(-expm1(-a))', linestyle=':', color='black', alpha=0.5)
#plt.plot(x_values, relative_error_log1p, label='log1p(-exp(-a))', linestyle='-.', color='red', alpha=0.5)
#plt.axhline(y=np.finfo(dtype).eps, color='black', linestyle='-', label=r'$\epsilon_c$', alpha=0.5)
#plt.axvline(x=np.log(2), color='gray', linestyle='--', label=r'$\log 2$')
plt.legend(fontsize=11)
plt.title(r'Relative error in computing $\log(1 - \exp(x))$', fontsize=13)
#plt.title(r'Computing $\log(1 - \exp(x))$: Relative Error', fontsize=13)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xlabel(r'$\log_2 |x|$', labelpad=-12)
#plt.ylabel('Relative Error')
plt.yscale('log', base=2)
plt.xscale('log', base=2)
xticks = [2.0**i for i in [-149, -125, -100, -50, -25, 0]]
xitcklabels = [f'${i}$' for i in [-149, -125, -100, -50, -25, 0]]
plt.xticks(xticks)
plt.gca().set_xticklabels(xitcklabels)
yticks = [2.0**i for i in [-35, -25, -15, -5]]
plt.yticks(yticks) #2.0**np.arange(-34, 1, 5))

plt.savefig(FIGURES_DIR + 'log1mexp.png', bbox_inches='tight', pad_inches=0.01, dpi=200)
plt.show()