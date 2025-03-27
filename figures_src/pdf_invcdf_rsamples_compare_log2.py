"""
Same as pdf_invcdf_rsamples_compare.py but with log2 conversion for a and b.
"""

import torch
from kumaraswamy import kumaraswamy_stable_log_pdf, kumaraswamy_stable_inverse_cdf, KumaraswamyStable
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Kumaraswamy as torch_kumaraswamy_dist
import math
log, log2 = torch.log, torch.log2
dtype = torch.float32
from config import FIGURES_DIR

log2_val = log(torch.tensor([2], dtype=dtype))

# specify in log2 base
log_b = torch.tensor(24)
#log_a_values = torch.tensor([3, 4, 5, 6], dtype=dtype) / log2_val
log_a_values = torch.tensor([2, 3, 4, 5], dtype=dtype) / log2_val

x = torch.linspace(0, 1, 50000)

density_y_min = 0#.5 # removes unaesthetic pdf lines at x = 0 
density_y_max = 80
x_lim_eps = .025
rsamples = 100000

# 4 subplots: top 3 for pytorch pdf and inverse_cdf and hist, bottom 3 for our pdf and inverse_cdf and hist
fig, axs = plt.subplots(2, 3, sharex=True, figsize=(10, 4)) #2.5)) # 4 for JOURNAL, 2.5 for CONFERENCE/ARXIV
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red'] 

## Pytorch
# pytorch pdf
for i, log_a in enumerate(log_a_values):
    y = torch_kumaraswamy_dist(log_a.exp2(), log_b.exp2()).log_prob(x).exp().detach().numpy()
    axs[0, 0].plot(x, y, color=colors[i])

# pytorch inverse_cdf
for i, log_a in enumerate(log_a_values):
    inverse_cdf = torch_kumaraswamy_dist(log_a.exp2(), log_b.exp2()).icdf(1-x).detach().numpy()
    axs[0, 1].plot(x, inverse_cdf, color=colors[i])

# reparameterized samples
pytorch_num_samples_underflow = []
for i, log_a in enumerate(log_a_values):
    samples = torch_kumaraswamy_dist(log_a.exp2(), log_b.exp2()).sample([rsamples]).flatten().detach().numpy()
    pytorch_num_samples_underflow.append((samples < 1e-10).sum())
    n, bins, patches = axs[0, 2].hist(samples, bins=100, range=(0, 1), color=colors[i]) #, zorder=len(log_a_values)-i)

print(f"PyTorch % samples underflow: {[f'{100 * x / rsamples:.2f}%' for x in pytorch_num_samples_underflow]}")


# the underflowing samples in the histogram overlap and are not visible. Manually draw
# a multi-color line from (0, 0) to (0, .39), using the colors of each histogram
# chop up the line into 8 segmentsa
first_bin_width = bins[1] - bins[0]  # Width is the difference between first two bin edges
first_bin_height = n[0]  # Height is the count in the first bi
x_start_first_bin = bins[0]
x_end_first_bin = bins[1]
middle_of_first_bin = (x_start_first_bin + x_end_first_bin) / 2

y_segments = np.linspace(0, first_bin_height, 9)
x_offset = first_bin_width / 2
for i, y in enumerate(y_segments[:-1]):
    axs[0, 2].plot([middle_of_first_bin, middle_of_first_bin], [y, y_segments[i+1]], color=colors[i % len(colors)], lw=1.6)# first_bin_width

# similarly, the all the inverse CDFs underflow beyond 1-39.3 making only the last one visible
# Draw a multi-color line from (1-.393, 0) to (1, 0), using the colors of each inverse CDF
x_segments = np.linspace(1 - .393, 1, 9)
y = 0
for i, segment_x in enumerate(x_segments[:-1]):
    axs[0, 1].plot([segment_x, x_segments[i+1]], [y, y], color=colors[i % len(colors)], lw=1.6)



# find largest x value where the inverse_cdf is > .01
inverse_cdf = torch_kumaraswamy_dist(log_a_values[-1].exp2(), log_b.exp2()).icdf(1-x).detach().numpy()
x_max = x[inverse_cdf > .01][-1]
percentage_underflow = 100 * (1 - x_max)
axs[0, 1].annotate(
    '', xy=(x_max, 0.4), xytext=(1, 0.4),
    arrowprops=dict(arrowstyle='<->', color='black', lw=1.5)
)
axs[0, 1].text((x_max + 1.02) / 2, 0.5, f'{percentage_underflow:.1f}% samples\nunderflow ', ha='center', fontsize=8)

print(f"Inverse CDF underflows at x={x_max:.3f}. Analytical expression predicts {1 - torch.exp(log_b - 25):.2f}")

## Custom ##

# convert to log base 2 for our implementation
log_a_values = log_a_values * log2_val
log_b = log_b * log2_val

# pdf
for i, log_a in enumerate(log_a_values):
    y = np.exp(kumaraswamy_stable_log_pdf(x, log_a, log_b).detach().numpy())
    axs[1, 0].plot(x, y, color=colors[i], label=f"{(log_a).item():.0f}")

# inverse_cdf
for i, log_a in enumerate(log_a_values):
    inverse_cdf = kumaraswamy_stable_inverse_cdf(x, log_a, log_b).detach().numpy()
    axs[1, 1].plot(x, inverse_cdf, color=colors[i], label=f"a=e^{log_a:.1f}, b=e^{log_b.item():.0f}")

# reparameterized samples
for i, log_a in enumerate(log_a_values):
    samples = KumaraswamyStable(log_a, log_b).sample([rsamples]).flatten().detach().numpy()
    axs[1, 2].hist(samples, bins=100, color=colors[i], range=(0, 1))


axs[0, 0].set_ylabel(r"PyTorch KS")
axs[1, 0].set_ylabel(r"Stable KS")
axs[1, 1].set_ylim(0, 1)
axs[0, 1].set_ylim(0, 1)
# remove yticks and ylabels from the leftmost column
for col in [0, 2]:
    axs[0, col].set_yticks([])
    axs[0, col].set_yticklabels([])
    axs[1, col].set_yticks([])
    axs[1, col].set_yticklabels([])
axs[0, 0].set_ylim(density_y_min, density_y_max)
axs[1, 0].set_ylim(density_y_min, density_y_max)

unit_ticks = [0,  1]
unit_tick_labels = [r'0', r'1']
for ax in axs.flatten():
    ax.set_xticks(unit_ticks)
    ax.set_xticklabels(unit_tick_labels)
    ax.set_xlim(-x_lim_eps, 1 + x_lim_eps)

axs[0, 1].set_yticks(unit_ticks)
axs[0, 1].set_yticklabels(unit_tick_labels)
axs[1, 1].set_yticks(unit_ticks)
axs[1, 1].set_yticklabels(unit_tick_labels)
axs[0, 1].set_ylim(-.02, 1)
axs[1, 1].set_ylim(-.02, 1)

axs[1, 0].legend(title=r"$\log a$", ncol=4, columnspacing=0.9, fontsize=8, title_fontsize=8)

axs[0, 0].set_title(r"PDF")#: \exp \log f\left(x\right)$")
axs[0, 1].set_title(r"Inverse CDF $F^{-1}(u)$")
axs[0, 2].set_title(r"Reparameterized Samples")
xpad = -15
axs[1, 0].set_xlabel(r'$x$', labelpad=xpad)  # 'x' with close padding
axs[1, 1].set_xlabel(r'$u$', labelpad=xpad)  # 'u' with close padding
axs[1, 2].set_xlabel(r'$x$', labelpad=xpad)  # 'x' with close padding

# manually make tight by setting distance between rows and columns
plt.subplots_adjust(wspace=0.1, hspace=0.125)

#plt.savefig(FIGURES_DIR + 'pdf_invcdf_rsamples_compare.png', bbox_inches='tight', pad_inches=0.01, dpi=200)
plt.show()