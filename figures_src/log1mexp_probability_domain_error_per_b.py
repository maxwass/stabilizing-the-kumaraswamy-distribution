"""
    Attempting to derive the probability of a domain error in log(1-exp(1/b * log u)), u \sim U(0,1) for each value of b
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions import Kumaraswamy
from config import FIGURES_DIR

dtype = np.float32

def log1mexp_method(a):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = a < math.log(2) # x < 0
        return np.where(
            mask,
            np.log(-np.expm1(-a, dtype=dtype), dtype=dtype),
            np.log1p(-np.exp(-a, dtype=dtype), dtype=dtype),
        )
    
def stable_log_prob(a, b, x):
    return np.log(a) + np.log(b) + (a-1) * np.log(x) + (b-1) * log1mexp_method(-1 * a * np.log(x))

"""
log(1 - exp(b log u)) faces domain error when -b log u < 2^-24 
    <--> u > exp(-b 2^-24), which has probability 1 - exp(-b 2^-24)
"""

b_values = np.logspace(1, 50, num=500, base=2)

prob_domain_error = 1 - np.exp(-b_values * 2**(-24))
prob_no_domain_error = 1 - prob_domain_error
num_samples_list = [1, 10, 100, 1000, 10000]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# First plot: Probability of domain error of the z
color_cycle1 = plt.cm.viridis(np.linspace(0, 1, len(num_samples_list)))
for color, num_samples in zip(color_cycle1, num_samples_list):
    # probability of at least one domain error 
    # = 1 - p(no domain errors in s samples) 
    # = 1 - p(no domain errors in 1 sample))^s, assuming independence of samples
    prob_domain_error_at_least_one = 1 - np.power(prob_no_domain_error, num_samples)
    ax1.plot(b_values, prob_domain_error_at_least_one, label=f'{num_samples}')
    
ax1.set_xscale('log', base=2)
ax1.set_xlabel('b', labelpad=-10, fontsize=14)
ax1.set_title(r'Probability of Domain Error: $\log(1-\exp(\frac{1}{b} \log u)), u \sim (0, 1)$', fontsize=14)
textstr = r'$p(\geq 1 \text{ error in } s \text{ samples})$' + \
    '\n' \
    + r'$= 1 - (1 - p(\text{error in } 1 \text{ sample}))^s$' + '\n'\
    + r'$= 1 - \exp(-b \cdot  2^{-24})^s$'
props = dict(boxstyle='round', facecolor='wheat', alpha=1)
ax1.text(0.475, 0.3, textstr, transform=ax1.transAxes, fontsize=14, 
         verticalalignment='top',
         bbox=props)
ax1.grid(True, which="both", ls="--")
ax1.set_xticks([2.0**i for i in range(0, 51, 10)])
ax1.legend(title=r'# samples (s)', loc='upper right', title_fontsize=14, 
           # set opacity
           framealpha=1.0)

# Second plot: Kumaraswamy distributions with mode at 0.5 with decreasing variance
### Interesting observation: \lim_{b_exp \to \infty} mode(Kumaraswamy(a, 2^b_exp)) = 0.5 when a = b_exp ##
b_exp_values =  np.array([1.0, 10.0, 20.0, 40.0]) 
a_values = [1.75, 10.1, 20.0, 40.0] 
color_cycle2 = plt.cm.tab10(np.linspace(.5, 1, len(b_exp_values)))
for color, (a, b_exp) in zip(color_cycle2, zip(a_values, b_exp_values)):
    b = 2.0**b_exp
    dist = Kumaraswamy(torch.tensor([a]), torch.tensor([b]))
    x = torch.linspace(0.000001, 1.0, 1000)
    y = stable_log_prob(a, b, x)
    ax2.plot(x, y.exp(), label=f'{b_exp:.0f}', color=color)
ax2.legend(title=r'$\log_2 b$', fontsize=14, title_fontsize=14, loc='upper right', framealpha=1.0)
xticks, xticklabels = [0, .25, .75, 1], ['0', '0.25', '0.75', '1']
ax2.set_xticks(xticks)
ax2.set_xticklabels(xticklabels)
ax2.set_yticks([0, 10, 20, 30])
ax2.set_title('Desirable Kumaraswamy Distributions With $b$ Large', fontsize=14)
ax2.set_xlabel('x', labelpad=-10, fontsize=14)
ax2.set_ylabel('Density', labelpad=-10, fontsize=14)

plt.tight_layout()
# force the subplots closer
plt.subplots_adjust(wspace=0.065)
#plt.savefig(FIGURES_DIR + 'prob_domain_error.png', bbox_inches='tight', pad_inches=0.01, dpi=200)
plt.show()