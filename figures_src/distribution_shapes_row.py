"""
    Comparing the canonical shapes of distributions
"""

import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import RelaxedBernoulli, Beta, ContinuousBernoulli

# for logistic normal
import tensorflow_probability as tfp
tfd = tfp.distributions

from kumaraswamy import KumaraswamyStable
from squashed_normal import TanhNormal
from config import FIGURES_DIR

dtype = torch.float32

fig, axs = plt.subplots(1, 4, figsize=(6, 2), sharey=True)
x = torch.linspace(0.0001, 0.9999, 5000, dtype=dtype)

""" Kumaraswamy """
ks_plot_idx = -1
ks_names = ["uniform", "bell", "decr.", "incr.", "U"]
ks_log_a = torch.tensor([0.0, 1, -1, 1, -20], dtype=dtype)
ks_log_b = torch.tensor([0.0, 1,  1, -1.5, -3], dtype=dtype)

for i in range(len(ks_log_a)):
    y = KumaraswamyStable(ks_log_a[i], ks_log_b[i]).log_prob(x).exp()
    axs[ks_plot_idx].plot(x, y, 
                   label=f"{ks_names[i]}: {ks_log_a[i]:.0f}, {ks_log_b[i]:.0f}",
                   linestyle="dotted")
ks_ylim = 2.3

""" Beta """
beta_plot_idx = -2
beta_names = ["uniform", "bell.", "incr.", "decr.", "U"]
beta_a = torch.tensor([1, 3,  .3, 5,  0.05], dtype=dtype) #torch.exp(ks_log_a)
beta_b = torch.tensor([1, 2.5, 5, .3, 0.05], dtype=dtype) #torch.exp(ks_log_b)
for i in range(len(beta_a)):
    y = Beta(beta_a[i], beta_b[i]).log_prob(x).exp()
    axs[beta_plot_idx].plot(x, y, 
                    label=f"{beta_names[i]}: {beta_a[i]:.1f}, {beta_b[i]:.1f}", 
                    linestyle="dotted")
beta_ylim = 2.3

""" Continuous Bernoulli """
cb_plot_idx = 0
#lambda_vals = [.001, .1, .5, .9, .999]
lambda_vals = [.01, .5, .99]
cb_colors = ["tab:green", "tab:blue", "tab:red"]
for i in range(len(lambda_vals)):
    y = ContinuousBernoulli(lambda_vals[i]).log_prob(x).exp()
    axs[cb_plot_idx].plot(x, y,
             label=f"{lambda_vals[i]:.3f}", 
             color=cb_colors[i],
             linestyle="dotted")
    
contin_bern_ylim = 2.5

""" Relaxed Bernoulli = Concrete """
conc_plot_idx = None
prob = torch.tensor([.01, .1, .5, .9]) #, .999])
#prob = torch.tensor([.05, .5, .9]) #, .999])
for i in range(len(prob)):
    y = RelaxedBernoulli(temperature=torch.tensor(.01), probs=prob[i]).log_prob(x).exp()
    #axs[conc_plot_idx].plot(x, y, label=f"{prob[i]:.3f}", linestyle="dotted")
conc_ylim = .05

""" Squashed Normal"""
logistic_or_tanh = "tanh"
ln_plot_idx = 1
# Define the parameters for the LogitNormal distribution
params = [
    (0.0, 1.78), # kinda uniform
    (0.1, 0.75), # bell
    (-1, 1.78), # left skewed = decr
    (1.0, 1.78),  # right skewed = incr
    (0.0, 10.0), # U-shaped
]
ln_names = ["uniform", "bell", "decr.", "incr.", "U"]
""" 
locs =   [0.00, 0.10, -3.00, 3.00, 0.00]
scales = [1.78, 0.80, 1.78,  1.78, 10.0]
for loc, scale in zip(locs, scales): #params:
    dist = tfd.LogitNormal(loc=loc, scale=scale)
    pdf_values = dist.prob(x)
    axs[ln_plot_idx].plot(x, pdf_values, linestyle="dotted")
"""
mus =      torch.tensor([0.0,  0.0, -2, 2, 0], dtype=dtype)
log_stdv = torch.tensor([-.1,  -.75, 0, 0, 2], dtype=dtype)
for mu, log_stdv in zip(mus, log_stdv):
    dist = TanhNormal(mu=mu, log_stdv=log_stdv, low=0, high=1)
    pdf_values = dist.log_prob(x).exp()
    axs[ln_plot_idx].plot(x, pdf_values, linestyle="dotted")


ln_ylim = 3

# set titles of each
#pad = -15
#axs[0, 0].set_title("KS", pad=pad)
#axs[1, 0].set_title("Beta", pad=pad)
#axs[0, 1].set_title('$\mathcal{CB}$', pad=pad) #"Contin. Bern.")
#axs[1, 1].set_title("Concrete", pad=pad) ##"Relaxed Ber.")
fontsize = 15
y_loc = 0.875
axs[ks_plot_idx].text(0.5, y_loc, "KS", ha='center', transform=axs[ks_plot_idx].transAxes, fontsize=fontsize)
axs[beta_plot_idx].text(0.5, y_loc, "Beta", ha='center', transform=axs[beta_plot_idx].transAxes, fontsize=fontsize)
axs[cb_plot_idx].text(0.5, y_loc, r'$\mathcal{CB}$', ha='center', transform=axs[cb_plot_idx].transAxes, fontsize=fontsize+2)
#axs[conc_plot_idx].text(0.5, y_loc, "Concrete", ha='center', transform=axs[conc_plot_idx].transAxes, fontsize=fontsize)
axs[ln_plot_idx].text(0.5, y_loc, r"$\tanh_{\mathcal{N}}$", ha='center', transform=axs[ln_plot_idx].transAxes, fontsize=fontsize)

# set y limits
""" 
axs[ks_plot_idx].set_ylim(0, ks_ylim)
axs[beta_plot_idx].set_ylim(0, beta_ylim)
axs[cb_plot_idx].set_ylim(0, contin_bern_ylim)
#axs[conc_plot_idx].set_ylim(0, conc_ylim)
axs[ln_plot_idx].set_ylim(0, ln_ylim)
"""
axs[ln_plot_idx].set_ylim(0, ln_ylim) # shared y, so affects all


# remove y ticks and labels from all subplots
for ax in axs.flatten():
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["0", "1"])

# Remove the top and right axis lines from all subplots
""" 
show = False
for i, ax in enumerate(axs.flatten()):
    ax.spines['top'].set_visible(show)
    ax.spines['right'].set_visible(show)
    ax.spines['left'].set_visible(show)
    ax.spines['bottom'].set_visible(show)
"""
    

plt.subplots_adjust(wspace=0.075, hspace=0.075)
#plt.savefig(FIGURES_DIR + 'distrib_shapes_row.png', bbox_inches='tight', pad_inches=0.01, dpi=200)
plt.show()