import torch
import numpy as np

from kumaraswamy import KumaraswamyStable
from vae import gauss_cb_VAE, gauss_beta_VAE, gauss_ks_VAE, ks_cb_VAE, ks_beta_VAE, ks_ks_VAE, beta_cb_VAE, beta_beta_VAE, beta_ks_VAE
from vae import MNISTDataModule

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import FIGURES_DIR

# load pytorch lightning model
from config import DATA_DIR, PROJECT_ROOT
path_to_project = PROJECT_ROOT + '/experiments/vae/'
config_path = path_to_project + 'config.yml'
path_to_models = path_to_project + 'trained_models/'


def load_model(variational_posterior, likelihood, dataset, path_to_models):
    model_path = path_to_models + f"{variational_posterior}_{likelihood}_{dataset}.ckpt"
    print(f"Loading: {model_path}")

    if variational_posterior == 'gaussian':
        if likelihood == 'cb':
            model_class = gauss_cb_VAE
        elif likelihood == 'beta':
            model_class = gauss_beta_VAE
        elif likelihood == 'ks':
            model_class = gauss_ks_VAE

    elif variational_posterior == 'ks':
        if likelihood == 'cb':
            model_class = ks_cb_VAE
        elif likelihood == 'beta':
            model_class = ks_beta_VAE
        elif likelihood == 'ks':
            model_class = ks_ks_VAE
    
    elif variational_posterior == 'beta':
        if likelihood == 'cb':
            model_class = beta_cb_VAE
        elif likelihood == 'beta':
            model_class = beta_beta_VAE
        elif likelihood == 'ks':
            model_class = beta_ks_VAE
    
    else:
        raise ValueError("Variational posterior must be one of ['gaussian', 'ks', 'beta']")


    model = model_class.load_from_checkpoint(model_path)
    return model

def likelihood_mean(likelihood_params, likelihood):
    if likelihood == 'cb':
        lambda_logit = likelihood_params
        x_hat_mean = torch.distributions.ContinuousBernoulli(lambda_logit).mean
    elif likelihood == 'beta':
        alpha, beta = likelihood_params
        x_hat_mean = torch.distributions.Beta(alpha, beta).mean
    elif likelihood == 'ks':
        log_a, log_b = likelihood_params
        x_hat_mean = KumaraswamyStable(log_a, log_b).sample((100,)).mean(dim=0) # mean function numerically unstable
    
    return x_hat_mean



## Load data and find first instance of each digit ##
dm = MNISTDataModule(batch_size=100, 
                     clamp_extreme_pixels=False, # will need to do this manually for beta/ks
                     data_dir=DATA_DIR, 
                     num_workers=0) # must be 0 in jupyter notebook

dm.prepare_data()
dm.setup('fit')
test_dataloader = dm.test_dataloader()

# Optional: Only show a subset of the digits
#which_digits = {}

# iterate through test data loader and find the first instance of each digit
first_instance = {}
for batch in test_dataloader:
    x, y = batch
    for i in range(len(y)):
        if y[i].item() not in first_instance:
            first_instance[y[i].item()] = x[i]
            if len(first_instance) == 10:
                break
    if len(first_instance) == 10:
        break

# create batch of images and labels
images = torch.stack(list(first_instance.values()))
labels = torch.tensor(list(first_instance.keys()))

# sort labels by digit
_, indices = torch.sort(labels)
images = images[indices]
labels = labels[indices]

first_instance_batch = (images, labels)

### Load models, and pass the first instance of each digit through them ##
dataset = 'mnist'
variational_posteriors = ['gaussian', 'ks', 'beta']
likelihoods = ['cb', 'beta', 'ks']
images = {f'{var_post}-{likelihood}': None for var_post in variational_posteriors for likelihood in likelihoods}
images['inputs'] = None

for likelihood in likelihoods:
    for variational_posterior in variational_posteriors:
        model = load_model(variational_posterior, likelihood, dataset, path_to_models)
        model.eval()
        with torch.no_grad():
            x, y = first_instance_batch
            x = x.view(x.size(0), -1).to(model.device)

            clamp_pixel_values = likelihood in ['beta', 'ks']
            if clamp_pixel_values: 
                x = torch.clamp(x, .5 * (1/255), 1 - (.5 * (1/255)))

            likelihood_params, _, _ = model(x)
            x_hat_mean = likelihood_mean(likelihood_params, likelihood)
            x_hat_mean = x_hat_mean.view(len(x), 28, 28)
            images[f'{variational_posterior}_{likelihood}'] = x_hat_mean
            images['inputs'] = x.view(len(x), 28, 28)



### Make the plot: Top Row Inputs, Subsequent rows, reconstructions ###


# Titles and labels for rows
image_name_to_row_title = {
    'Inputs': 'Inputs',
    # CB
    "gaussian_cb": r"$\mathcal{N}$-$\mathcal{CB}$",
    "ks_cb": r"KS-$\mathcal{CB}$",
    "beta_cb": r"Beta-$\mathcal{CB}$",
    # Beta
    "gaussian_beta": r"$\mathcal{N}$-Beta",
    "ks_beta": r"KS-Beta",
    "beta_beta": r"Beta-Beta",
    # KS
    "gaussian_ks": r"$\mathcal{N}$-KS",
    "ks_ks": r"KS-KS",
    "beta_ks": r"Beta-KS",
}
# Define the plot with GridSpec
#fig = plt.figure(figsize=(3.33, 2.333))#(10, 7))
#gs = gridspec.GridSpec(7, 10, wspace=0.0, hspace=0.0)
fig = plt.figure(figsize=(3.1, 2.333)) #(10, 9))
gs = gridspec.GridSpec(len(image_name_to_row_title), 10, wspace=0, hspace=0) #0.0)


# Add row titles
for i, title_key in enumerate(image_name_to_row_title):
    ax = fig.add_subplot(gs[i, 0])
    ax.set_ylabel(image_name_to_row_title[title_key], rotation=0, fontsize=8, labelpad=2, ha='right', va='center')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

# Plot images
#for row in range(len(images)):
for row, title_key in enumerate(image_name_to_row_title):
    for col in range(10):
        ax = fig.add_subplot(gs[row, col])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        if row == 0:
            ax.imshow(first_instance_batch[0][col].cpu().numpy().reshape(28, 28), cmap='gray')
        else:
            #key = row_titles[row].replace("-", "_").replace(r"$\mathcal{N}$", "gaussian").lower()
            #ax.imshow(images[key][col].cpu().numpy(), cmap='gray')
            ax.imshow(images[title_key][col].cpu().numpy(), cmap='gray', vmin=0, vmax=1)

plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
#plt.subplots_adjust(wspace=0.5, hspace=0.0) # this removes ALL white space

#plt.savefig(FIGURES_DIR + 'mnist_reconstructions_grid.png', bbox_inches='tight', pad_inches=0.01, dpi=250)
plt.show()