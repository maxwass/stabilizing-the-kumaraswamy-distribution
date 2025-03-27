import torch
import numpy as np
import matplotlib.pyplot as plt
from kumaraswamy import KumaraswamyStable
from torch.distributions.utils import euler_constant
from config import FIGURES_DIR

def kumaraswamy_stable_entropy(log_a, log_b):
        t1 = 1 - torch.exp(- log_a)
        t0 = 1 - torch.exp(- log_b)
        b = torch.exp(log_b)
        H0 = torch.digamma(b + 1) + euler_constant
        return  t0 + t1 * H0 - log_b - log_a
        """
        t1 = 1 - self.concentration1.reciprocal()
        t0 = 1 - self.concentration0.reciprocal()
        H0 = torch.digamma(self.concentration0 + 1) + euler_constant
        return (
            t0
            + t1 * H0
            - torch.log(self.concentration1)
            - torch.log(self.concentration0)
        )
        """

def kumaraswamy_mean(log_a, log_b):
        b = torch.exp(log_b)
        a_inv = torch.exp(- log_a)
        mean = (log_b + torch.special.gammaln(1 + a_inv) + torch.special.gammaln(b) - torch.special.gammaln(1 + a_inv + b)).clamp(1e-5, 1).exp()
        return mean


fontsize = 16

# Generate a range of values for a and b
dtype = torch.float32
lim = -20
log_a = torch.linspace(-10, 30, 1600)  # 3000 Smaller range for visibility
log_b = torch.linspace(-10, 30, 1600)
log_a_m, log_b_m = torch.meshgrid(log_a, log_b, indexing='ij')

####### ENTROPY #######
#entropy = - kumaraswamy_stable_entropy(torch.exp(log_a_m), torch.exp(log_b_m))
entropy = - kumaraswamy_stable_entropy(log_a_m, log_b_m)

# Create the plot with final adjustments
fig, ax = plt.subplots(figsize=(8, 6))
# plot log - entropy
vmin, vmax = -2, 5#1.25 #-1.5, 1.5
extent = (-lim, lim, -lim, lim)

levels = np.linspace(vmin, vmax, num=200)  # 200 levels from vmin to vmax
cp = ax.contourf(log_a_m, log_b_m, torch.log(entropy).detach().numpy(), levels=levels, cmap='viridis', extend='both')
cbar = plt.colorbar(cp, ticks=np.arange(vmin, vmax + 0.5, 1), pad=0.01)  # Adjust the ticks if necessary
#cbar.set_label('Entropy (log scale)', rotation=270, labelpad=20)

cbar.ax.yaxis.set_label_position('right')
cbar.ax.yaxis.tick_right()

ax.set_title(r'Differential Entropy ($\log$)', fontsize=fontsize) # $\log_{10}-H_{\alpha, \beta}$', fontsize=fontsize)

# Adjust labels
ax.set_xlabel(r"$\log$ {0}".format(r"a"), fontsize=fontsize)
ax.set_ylabel(r'$\log$ {0}'.format(r"b"), fontsize=fontsize)
ax.xaxis.labelpad = -2
ax.yaxis.labelpad = -6

# Highlighting points where the mean is ~0.5
mask = torch.isclose(entropy, torch.tensor(1.0, dtype=dtype), atol=0.01)
highlighted_points = ax.scatter(log_a_m[mask], log_b_m[mask], color='black', 
                                label=r'1.0', s=2)
ax.legend(loc=(.8, 0.75), fontsize=12)

# Adding reference lines and text
ax.axhline(0, color='grey', linestyle='--')
ax.axvline(0, color='grey', linestyle='--')

# Adjust text annotations to avoid overlapping with the legend
text_props = dict(boxstyle='round', facecolor='white', alpha=1.0)
ax.text(0.95, 0.95, 'Bell', transform=ax.transAxes, verticalalignment='top', horizontalalignment='right', bbox=text_props, fontsize=fontsize)
ax.text(0.05, 0.95, 'Decr.', transform=ax.transAxes, verticalalignment='top', horizontalalignment='left', bbox=text_props, fontsize=fontsize)
ax.text(0.05, 0.05, 'U', transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='left', bbox=text_props, fontsize=fontsize)
ax.text(0.95, 0.05, 'Incr.', transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='right', bbox=text_props, fontsize=fontsize)

plt.savefig(FIGURES_DIR + 'entropy.png', bbox_inches='tight', pad_inches=0, dpi=400)
#plt.show()

###### MEAN #######


# set seed
torch.manual_seed(50)
# mean
mean = KumaraswamyStable(log_a_m, log_b_m).sample((500,)).mean(dim=0)

# Create the plot with final adjustments
fig, ax = plt.subplots(figsize=(8, 6))
# plot log - entropy
vmin, vmax = 0, 1
extent = (-lim, lim, -lim, lim)

levels = np.linspace(vmin, vmax, num=200)  # 200 levels from vmin to vmax
cp = ax.contourf(log_a_m, log_b_m, mean.detach().numpy(), levels=levels, cmap='viridis', extend='both')
cbar = plt.colorbar(cp, ticks=np.linspace(0, 1, 6), pad=0.01)  # Adjust the ticks if necessary
#cbar.set_label('Entropy (log scale)', rotation=270, labelpad=20)

cbar.ax.yaxis.set_label_position('right')
cbar.ax.yaxis.tick_right()

ax.set_title(r'Mean', fontsize=fontsize) # $\log_{10}-H_{\alpha, \beta}$', fontsize=fontsize)

# Adjust labels
ax.set_xlabel(r"$\log$ {0}".format(r"a"), fontsize=fontsize)
ax.set_ylabel(r'$\log$ {0}'.format(r"b"), fontsize=fontsize)
ax.xaxis.labelpad = -2
ax.yaxis.labelpad = -6

# Highlighting points where the mean is ~0.5
mask = torch.isclose(mean, torch.tensor(0.5, dtype=dtype), atol=0.01)
highlighted_points = ax.scatter(log_a_m[mask], log_b_m[mask], color='black', 
                                label=r'1.0', s=2)
ax.legend(loc=(.8, 0.75), fontsize=12)

# Adding reference lines and text
ax.axhline(0, color='grey', linestyle='--')
ax.axvline(0, color='grey', linestyle='--')

# Adjust text annotations to avoid overlapping with the legend
text_props = dict(boxstyle='round', facecolor='white', alpha=1.0)
ax.text(0.95, 0.95, 'Bell', transform=ax.transAxes, verticalalignment='top', horizontalalignment='right', bbox=text_props, fontsize=fontsize)
ax.text(0.05, 0.95, 'Decr.', transform=ax.transAxes, verticalalignment='top', horizontalalignment='left', bbox=text_props, fontsize=fontsize)
ax.text(0.05, 0.05, 'U', transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='left', bbox=text_props, fontsize=fontsize)
ax.text(0.95, 0.05, 'Incr.', transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='right', bbox=text_props, fontsize=fontsize)
#plt.savefig(FIGURES_DIR + 'mean.png', bbox_inches='tight', pad_inches=0, dpi=400)
plt.show()
