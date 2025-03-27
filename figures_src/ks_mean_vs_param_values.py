import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from config import FIGURES_DIR


def kumaraswamy_moment(a, b, n):
    gamma_1_a = torch.lgamma(1 + n / a)  # Gamma(1 + n/a)
    gamma_b = torch.lgamma(b)            # Gamma(b)
    gamma_1_a_b = torch.lgamma(1 + n / a + b)  # Gamma(1 + n/a + b)

    moment = torch.exp(torch.log(b) + gamma_1_a + gamma_b - gamma_1_a_b)
    return moment

def kumaraswamy_mean(a, b):
    return kumaraswamy_moment(a, b, 1)
fontsize = 12

# Generate a range of values for a and b
dtype = torch.float64
#log_a = torch.linspace(-5, 5, 2000, dtype=dtype)  # Smaller range for visibility
#log_b = torch.linspace(-5, 5, 2000, dtype=dtype)
lim = 10
log_a = torch.linspace(-lim, 6, 2000, dtype=dtype)  # Smaller range for visibility
log_b = torch.linspace(-6, lim, 2000, dtype=dtype)
log_a_m, log_b_m = torch.meshgrid(log_a, log_b, indexing='ij')

# Initialize a grid for storing mean values
mean_values = kumaraswamy_mean(torch.exp(log_a_m), torch.exp(log_b_m))
mean_values = mean_values.clamp(0, 1)

# Create the plot with final adjustments
fig, ax = plt.subplots(figsize=(4,4)) #8, 6))
cp = ax.contourf(log_a_m, log_b_m, mean_values.numpy(), levels=100, cmap='viridis', vmin=0, vmax=1)
cbar = plt.colorbar(cp, ticks=np.arange(0, 1.1, 0.2), pad=0.01)
#cbar.ax.set_ylabel('Mean Value')
cbar.ax.yaxis.set_label_position('right')
cbar.ax.yaxis.tick_right()
plt.title('KS Mean', fontsize=fontsize-2)

# manually set ticks
#ticks = [-4, -3, -2, -1, 1, 2, 3, 4]
ticks = [-6, -4, -2, 2, 4, 6]
ax.set_xticks(ticks)
ax.set_yticks(ticks)
# set font size of ticks
ax.tick_params(axis='both', which='major', labelsize=fontsize-2)

# Adjust labels
ax.set_xlabel(r"$\log$ {0}".format(r"a"), fontsize=fontsize+1)
ax.set_ylabel(r'$\log$ {0}'.format(r"b"), fontsize=fontsize+1)
ax.xaxis.labelpad = -11
ax.yaxis.labelpad = -14

# Highlighting points where the mean is ~0.5
mask = torch.isclose(mean_values, torch.tensor(0.5, dtype=dtype), atol=0.01)
highlighted_points = ax.scatter(log_a_m[mask], log_b_m[mask], color='fuchsia', label='0.5', s=2)
# choose every 500th point to print
# print(log_a_m[mask][::500], log_b_m[mask][::500])

# Define the fitted function using the parameters obtained
#def fitted_curve(log_a, k1, k2, c):
#    return k1 * np.exp(k2 * log_a) + c
#k1, k2, c = 1.7962, .6890, -2.1731
#log_b_fit =  fitted_curve(log_a, k1, k2, c)

def modified_fitted_curve(log_a, k1, k2, c):
    return k1 * np.exp(k2 * log_a) * np.log(2) + c
# Fit the data using the modified function to find the best-fit parameters
params_modified, _ = curve_fit(modified_fitted_curve, log_a_m[mask], log_b_m[mask])
k1, k2, c = params_modified
print(f"Modified fitted curve: \n\tk1={k1:.5f}, \n\tk2={k2:.5f}, \n\tc={c:.5f}")
log_b_fit = modified_fitted_curve(log_a, k1, k2, c)


# Plot the fitted curve
ax.plot(log_a, log_b_fit, color='red', linestyle='--', linewidth=2, label='Fitted Curve')

# constrain y-axis to be between -6 and 6
ax.set_ylim(-lim, lim)

# add legend to the middle right (inside the plot but around .75 y value), and set font size
# manually set position of legend to avoid overlapping with highlighted points
ax.legend(loc=(.65, 0.65), fontsize=fontsize-2)

# Adding reference lines and text
ax.axhline(0, color='grey', linestyle='--')
ax.axvline(0, color='grey', linestyle='--')

# Adjust text annotations to avoid overlapping with the legend
text_props = dict(boxstyle='round', facecolor='white', alpha=1.0)
ax.text(0.95, 0.95, 'Bell', transform=ax.transAxes, verticalalignment='top', horizontalalignment='right', bbox=text_props, fontsize=fontsize-2)
ax.text(0.05, 0.95, 'Decreasing', transform=ax.transAxes, verticalalignment='top', horizontalalignment='left', bbox=text_props, fontsize=fontsize-2)
ax.text(0.05, 0.05, 'U', transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='left', bbox=text_props, fontsize=fontsize-2)
ax.text(0.95, 0.05, 'Increasing', transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='right', bbox=text_props, fontsize=fontsize-2)

# add the following line
# y = exp(x) if x >= 0, -log(-x) if x < 0.
#log_b_approx = torch.where(log_a >= 0, torch.exp(log_a), -torch.log(-log_a))
#ax.plot(log_a, log_b_approx, color='red', label='Approx', linewidth=1.5)
#ax.legend(loc=(.65, 0.65), fontsize=fontsize-2)

#plt.savefig(FIGURES_DIR + 'square_mean.png', bbox_inches='tight', pad_inches=0, dpi=300)
plt.show()