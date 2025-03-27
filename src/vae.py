import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10

import lightning as L

import matplotlib.pyplot as plt
from matplotlib import colors

import wandb
import numpy as np

from kumaraswamy import KumaraswamyStable


#### DATA #####

def clamp_pixel_values(x):
    # clamp the the middle of the smallest/largest quantized bucket
    # e.g. for 8 bit images, where pixels are in {0, 1, ..., 255}, clamp to [.5, 254.5] / 255
    dtype = x.dtype
    device = x.device
    #smallest_subnormal = torch.tensor(1.401298464324817e-45, dtype=dtype, device=device) if dtype == torch.float32 else torch.tensor(4.9406564584124654e-324, dtype=dtype, device=device)
    #largest_less_than_one = torch.tensor(1 - 2**(-24), dtype=dtype, device=device)  #if dtype == torch.float32 else torch.tensor(1 - 2**-53, dtype=dtype, device=device)
    center_smallest_quantile = torch.tensor(.5 * (1/255), dtype=dtype, device=device) #if dtype == torch.float32 else torch.tensor(2**-53, dtype=dtype, device=device)
    center_largest_quantile = torch.tensor(1 - .5 * (1/255), dtype=dtype, device=device) 
        
    return x.clamp(min=center_smallest_quantile, max=center_largest_quantile)

class MNISTDataModule(L.LightningDataModule):
    def __init__(self, batch_size: int, clamp_extreme_pixels: bool, data_dir: str = "../data/", num_workers: int = 0):
        # from tutorial page: https://lightning.ai/docs/pytorch/stable/data/datamodule.html
        super().__init__()
        self.data_dir = data_dir
        if clamp_extreme_pixels:
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1)), clamp_pixel_values])
        else:
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
        
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            #mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            #self.mnist_train, self.mnist_val = random_split(
            #    mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
            #)
            self.mnist_train = MNIST(self.data_dir, train=True, transform=self.transform)

        # Assign test dataset for use in dataloader(s)
        #if stage == "test":
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

        if stage == "predict":
            self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        # USING TEST SET FOR VALIDATION - only using validation for debugging/viz
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size, num_workers=self.num_workers)
    
class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self, batch_size: int, clamp_extreme_pixels: bool, data_dir: str = "../data/", num_workers: int = 0):
        # from tutorial page: https://lightning.ai/docs/pytorch/stable/data/datamodule.html
        super().__init__()
        self.data_dir = data_dir
        if clamp_extreme_pixels:
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1)), clamp_pixel_values])
        else:
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
        
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        # download
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            #mnist_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            #self.mnist_train, self.mnist_val = random_split(
            #    mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
            #)
            self.cifar10_train = CIFAR10(self.data_dir, train=True, transform=self.transform)

        # Assign test dataset for use in dataloader(s)
        #if stage == "test":
            self.cifar10_test = CIFAR10(self.data_dir, train=False, transform=self.transform)

        if stage == "predict":
            self.cifar10_predict = CIFAR10(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.cifar10_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        # USING TEST SET FOR VALIDATION - only using validation for debugging/viz
        return DataLoader(self.cifar10_test, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.cifar10_test, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size, num_workers=self.num_workers)
    

#### MODELING #####

def kl_standard_normal(mu, log_var):
    # closed form expression for KL divergence between vector of scalar gaussians and standard normal
    # see https://arxiv.org/pdf/1312.6114, pg 5 bottom
    # see https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py, loss_function()
    return - 0.5 * ( 1 + log_var - mu ** 2 - log_var.exp())

def sample_images(model, batch, num_images=10):
    model.eval()
    #batch = next(iter(loader))
    x, _ = batch
    x = x.view(x.size(0), -1)
    x = x.to(model.device)
    with torch.no_grad():
        likelihood_params, _, _ = model(x)
    return x, likelihood_params

def encoder_mnist(input_dim, hidden_dim, latent_dim, keep_prob, n_output):
    enc = nn.Sequential(
        nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(), nn.Dropout(1-keep_prob),
        nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(), nn.Dropout(1-keep_prob),
        #nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(1-keep_prob),# added
        #nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(1-keep_prob),# adde
        #nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Dropout(1-keep_prob), # add in for more params
        nn.Linear(hidden_dim, latent_dim * n_output)  # mean and log variance
        )
    return enc

def decoder_mnist(input_dim, hidden_dim, latent_dim, keep_prob, n_output):
    dec = nn.Sequential(
        nn.Linear(latent_dim, hidden_dim), nn.LeakyReLU(), nn.Dropout(1-keep_prob),
        nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(), nn.Dropout(1-keep_prob),
        #nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(1-keep_prob), # added
        #nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(1-keep_prob), # added
        #nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Dropout(1-keep_prob), # add in for more params
        nn.Linear(hidden_dim, n_output * input_dim),
        )
    return dec

class EncoderCIFAR(nn.Module):
    def __init__(self, n_output: int, latent_dim: int, feat: int = 32):
        super(EncoderCIFAR, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=2, padding=0, stride=1) #  out: 3 x 31 x 31
        self.conv2 = nn.Conv2d(3, feat, kernel_size=2, padding=1, stride=2)  # out: 32 x 16 x 16
        self.conv3 = nn.Conv2d(feat, feat, kernel_size=3, padding=1) # out: 32 x 16 x 16
        self.conv4 = nn.Conv2d(feat, feat, kernel_size=3, padding=1) # out: 32 x 16 x 16
        self.fc1 = nn.Linear(feat * 16 * 16, 128) # out: 128
        self.fc2 = nn.Linear(128, n_output * latent_dim) # out: n_output * latent_dim
    
    def forward(self, x):
        x = x.view(-1, 3, 32, 32)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        return self.fc2(x)
    
class DecoderCIFAR(nn.Module):
    def __init__(self, n_output, latent_dim, feat=32):
        super(DecoderCIFAR, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, feat * 16 * 16)
        self.tconv1 = nn.ConvTranspose2d(feat, feat, kernel_size=3, padding=1)
        self.tconv2 = nn.ConvTranspose2d(feat, feat, kernel_size=3, padding=1)
        self.tconv3 = nn.ConvTranspose2d(feat, feat, kernel_size=3, stride=2, padding=1, output_padding=1)
        # last conv1 output shape should be: n_output * 3 x 32 x 32
        self.conv1 = nn.Conv2d(feat, n_output * 3, kernel_size=3, padding=1)

    def forward(self, z):
        x = F.leaky_relu(self.fc1(z))
        x = F.leaky_relu(self.fc2(x))
        x = x.view(x.size(0), -1, 16, 16)
        x = F.leaky_relu(self.tconv1(x))
        x = F.leaky_relu(self.tconv2(x))
        x = F.leaky_relu(self.tconv3(x))
        return self.conv1(x).view(x.size(0), -1)


#### Gaussian Variational Posterior ####

class gauss_cb_VAE(L.LightningModule):
    def __init__(self,
                 hidden_dim=500, 
                 latent_dim=20, 
                 keep_prob=0.9,
                 dataset='mnist',
                 learning_rate=1e-3
                 ):
        super(gauss_cb_VAE, self).__init__()

        assert dataset in ['mnist', 'cifar10'], f'unrecognized dataset: {dataset}'
        self.save_hyperparameters()

        self.n_output_encoder = 2 # gaussian
        self.n_output_decoder = 1 # continuous bernoulli
        if dataset == 'mnist':
            self.encoder = encoder_mnist(input_dim=int(28*28), hidden_dim=hidden_dim, latent_dim=latent_dim, keep_prob=keep_prob, n_output=self.n_output_encoder)
            self.decoder = decoder_mnist(input_dim=int(28*28), hidden_dim=hidden_dim, latent_dim=latent_dim, keep_prob=keep_prob, n_output=self.n_output_decoder)
        else:
            self.encoder = EncoderCIFAR(
                n_output=self.n_output_encoder, # for gaussian, KS, beta, etc
                latent_dim=latent_dim)
            self.decoder = DecoderCIFAR(n_output=self.n_output_decoder, latent_dim=latent_dim)
        # Initialize lists to store outputs for logging
        # https://lightning.ai/releases/2.0.0/#bc-changes-pytorch
        self.train_outputs = []
        self.val_outputs = []
        self.test_outputs = []

    ## START: Common to ALL VAEs ##   
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, 'train', self.train_outputs)
    
    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            fig = self.visualize_reconstructions(train_or_test='test', num_images=10)
            self.logger.experiment.log({"test reconstructions": wandb.Image(fig)})
            plt.close(fig)

            fig = self.visualize_reconstructions(train_or_test='train', num_images=10)
            self.logger.experiment.log({"train reconstructions": wandb.Image(fig)})
            plt.close(fig)
        return self.shared_step(batch, batch_idx, 'val', self.val_outputs)
    
    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, 'test', self.test_outputs)
    
    def shared_epoch_end(self, stage, stage_outputs):
        # Calculate average metrics
        avg_elbo = torch.stack([x['elbo'] for x in stage_outputs]).mean()
        avg_log_prob_data = torch.stack([x['log_prob_data'] for x in stage_outputs]).mean()
        avg_kl = torch.stack([x['kl'] for x in stage_outputs]).mean()
        
        # Log the average metrics
        self.log(f'avg_elbo_{stage}', avg_elbo)
        self.log(f'avg_log_prob_data_{stage}', avg_log_prob_data)
        self.log(f'avg_kl_{stage}', avg_kl)
        
        # Clear the outputs for the next epoch
        stage_outputs.clear()

    def on_train_epoch_end(self):
        self.shared_epoch_end('train', self.train_outputs)
    
    def on_validation_epoch_end(self):
        self.shared_epoch_end('val', self.val_outputs)

    def on_test_epoch_end(self):
        self.shared_epoch_end('test', self.test_outputs)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
    
    ## END: Common to ALL VAEs ##
    

    ## START: Common to all Gaussian Variational Posterior models ##

    def encode(self, x):
        x_encoded = self.encoder(x)
        mu, log_var = x_encoded.chunk(2, dim=-1)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return z
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var
    
    ## END: Common to all Gaussian Variational Posterior models ##

    ## START: Common to CB likelihood ##

    def decode(self, z):
        lambda_logit = self.decoder(z)
        return lambda_logit

    def visualize_reconstructions(self, train_or_test, num_images=10):
        # sample images
        self.eval()
        loader = self.trainer.datamodule.train_dataloader() if train_or_test == 'train' else self.trainer.val_dataloaders
        x, _ = next(iter(loader)) # first batch
        x = x.view(x.size(0), -1)
        x = x.to(self.device)
        with torch.no_grad():
            likelihood_params, _, _ = self(x)

        # reshape
        size = 28 if self.hparams.dataset == 'mnist' else 32
        channels = 1 if self.hparams.dataset == 'mnist' else 3
        likelihood_params = likelihood_params.view(-1, channels, size, size)
        x = x.view(-1, channels, size, size)

        x_hat_mean = torch.distributions.ContinuousBernoulli(logits=likelihood_params).mean

        x = x.cpu()
        x_hat_mean = x_hat_mean.cpu()

        fig, axes = plt.subplots(2, num_images, figsize=(num_images, 2))
        for i in range(num_images):
            if self.hparams.dataset == 'mnist':
                axes[0, i].imshow(x[i].squeeze(), cmap='gray', vmin=0, vmax=1)
                axes[0, i].axis('off')
                axes[1, i].imshow(x_hat_mean[i].squeeze(), cmap='gray', vmin=0, vmax=1)
                axes[1, i].axis('off')
            else:
                # cifar10
                img = np.transpose(x[i].numpy(), (1, 2, 0))
                img_hat = np.transpose(x_hat_mean[i].numpy(), (1, 2, 0))
                axes[0, i].imshow(img)
                axes[0, i].axis('off')
                axes[1, i].imshow(img_hat)
                axes[1, i].axis('off')
        return fig

    ## END: Common to CB likelihood ##


    ## START: Specific to Gaussian Variational Posterior AND CB likelihood ##

    def shared_step(self, batch, batch_idx, stage, stage_outputs):
        x, _ = batch

        lambda_logit, mu, log_var = self(x)

        lambda_logit = lambda_logit.view(x.size(0), -1)
        x = x.view(x.size(0), -1)
        
        log_prob_data = torch.distributions.ContinuousBernoulli(logits=lambda_logit).log_prob(x).sum(-1)

        kl = kl_standard_normal(mu, log_var).sum(-1)

        elbo = log_prob_data - kl
        neg_elbo = -elbo
        neg_elbo_mean = neg_elbo.mean()
        self.log(f'elbo_{stage}', -neg_elbo_mean, prog_bar=True)
        self.log(f'kl_{stage}', kl.mean())
        self.log(f'log_prob_data_{stage}', log_prob_data.mean())
         
        stage_outputs.append({'elbo': elbo.detach(), 'log_prob_data': log_prob_data.detach(), 'kl': kl.detach()})

        return neg_elbo_mean


    def shared_step_iwae(self, batch, batch_idx, stage, stage_outputs, k=200):
        x, _ = batch

        def reparameterize(mu, log_var, k):
            std = torch.exp(log_var / 2)
            q = torch.distributions.Normal(mu, std)
            z = q.rsample((k,))
            return z

        
        mu, log_var = self.encode(x)
        z = reparameterize(mu, log_var, k)
        # decode
        ### for MNIST
        #lambda_logit = self.decoder(z) # CHANGED FOR CIFAR10
        ### for CIFAR10
        lambda_logit = self.decode(z.view(-1, z.shape[-1]))  # Fixed line
        # Reshape lambda_logit back to (k, batch_size, num_pixels)
        lambda_logit = lambda_logit.view(k, x.size(0), -1)
        

        # ELBO \approx \frac{1}{K} \sum_{k=1}^K ( log p(x | z_k) + log p(z_k) - log q(z_k | x) ), z_k \sim q(z|x)
        log_likelihood = torch.distributions.ContinuousBernoulli(logits=lambda_logit).log_prob(x).sum(-1)
        log_prior = torch.distributions.Normal(loc=0, scale=1).log_prob(z).sum(-1) # std normal prior
        log_var_post =  torch.distributions.Normal(loc=mu, scale=torch.exp(log_var / 2)).log_prob(z).sum(-1) # encoded var post

        # shape: (k, batch)
        log_importance_weights = log_likelihood + log_prior - log_var_post
        
        importance_weights = torch.logsumexp(log_importance_weights, 0) # one weight per sample in 

        elbo_estimate_batch = importance_weights.mean() # iwae elbo estimate for EACH image sample
        kl = kl_standard_normal(mu, log_var).sum(-1)

        stage_outputs.append({'elbo': importance_weights.detach(),
                              'log_prob_data': log_likelihood.detach(), 
                              'kl': kl.detach()
                              })

        return importance_weights


    ## END: Specific to Gaussian Variational Posterior AND CB likelihood ##

class gauss_beta_VAE(L.LightningModule):
    def __init__(self, 
                 hidden_dim=500, 
                 latent_dim=20, 
                 keep_prob=0.9,
                 dataset='mnist',
                 learning_rate=1e-3
                 ):
        super(gauss_beta_VAE, self).__init__()

        assert dataset in ['mnist', 'cifar10'], f'unrecognized dataset: {dataset}'
        self.save_hyperparameters()

        self.n_output_encoder = 2 # gaussian
        self.n_output_decoder = 2 # beta
        if dataset == 'mnist':
            self.encoder = encoder_mnist(input_dim=int(28*28), hidden_dim=hidden_dim, latent_dim=latent_dim, keep_prob=keep_prob, n_output=self.n_output_encoder)
            self.decoder = decoder_mnist(input_dim=int(28*28), hidden_dim=hidden_dim, latent_dim=latent_dim, keep_prob=keep_prob, n_output=self.n_output_decoder)
        else:
            self.encoder = EncoderCIFAR(
                n_output=self.n_output_encoder, # for gaussian, KS, beta, etc
                latent_dim=latent_dim)
            self.decoder = DecoderCIFAR(n_output=self.n_output_decoder, latent_dim=latent_dim)
        # Initialize lists to store outputs for logging
        # https://lightning.ai/releases/2.0.0/#bc-changes-pytorch
        self.train_outputs = []
        self.val_outputs = []
        self.test_outputs = []
    
    ## START: Common to ALL VAEs ##

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, 'train', self.train_outputs)
    
    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            fig = self.visualize_reconstructions(train_or_test='test', num_images=10)
            self.logger.experiment.log({"test reconstructions": wandb.Image(fig)})
            plt.close(fig)

            fig = self.visualize_reconstructions(train_or_test='train', num_images=10)
            self.logger.experiment.log({"train reconstructions": wandb.Image(fig)})
            plt.close(fig)
        return self.shared_step(batch, batch_idx, 'val', self.val_outputs)
    
    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, 'test', self.test_outputs)
    
    def shared_epoch_end(self, stage, stage_outputs):
        # Calculate average metrics
        avg_elbo = torch.stack([x['elbo'] for x in stage_outputs]).mean()
        avg_log_prob_data = torch.stack([x['log_prob_data'] for x in stage_outputs]).mean()
        avg_kl = torch.stack([x['kl'] for x in stage_outputs]).mean()
        
        # Log the average metrics
        self.log(f'avg_elbo_{stage}', avg_elbo)
        self.log(f'avg_log_prob_data_{stage}', avg_log_prob_data)
        self.log(f'avg_kl_{stage}', avg_kl)
        
        # Clear the outputs for the next epoch
        stage_outputs.clear()

    def on_train_epoch_end(self):
        self.shared_epoch_end('train', self.train_outputs)
    
    def on_validation_epoch_end(self):
        self.shared_epoch_end('val', self.val_outputs)

    def on_test_epoch_end(self):
        self.shared_epoch_end('test', self.test_outputs)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
    
    ## END: Common to ALL VAEs ##


    ## START: Common to all Gaussian Variational Posterior models ##
    
    def encode(self, x):
        x_encoded = self.encoder(x)
        mu, log_var = x_encoded.chunk(2, dim=-1)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return z
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    ## END: Common to all Gaussian Variational Posterior models ##


    ## START: Common to Beta likelihood ##

    def decode(self, z):
        likelihood_params = self.decoder(z)
        log_alpha, log_beta = likelihood_params.chunk(2, dim=-1)
        alpha, beta = 1e-6 + torch.nn.functional.softplus(log_alpha), 1e-6 + torch.nn.functional.softplus(log_beta)
        return (alpha, beta)

    def visualize_reconstructions(self, train_or_test, num_images=10):
        # sample images
        self.eval()
        loader = self.trainer.datamodule.train_dataloader() if train_or_test == 'train' else self.trainer.val_dataloaders
        x, _ = next(iter(loader)) # first batch
        x = x.view(x.size(0), -1)
        x = x.to(self.device)
        with torch.no_grad():
            likelihood_params, _, _ = self(x)
        alpha, beta = likelihood_params

        # reshape
        size = 28 if self.hparams.dataset == 'mnist' else 32
        channels = 1 if self.hparams.dataset == 'mnist' else 3
        alpha = alpha.view(-1, channels, size, size)
        beta = beta.view(-1, channels, size, size)
        x = x.view(-1, channels, size, size)

        x_hat_mean = torch.distributions.Beta(alpha, beta).mean

        x = x.cpu()
        x_hat_mean = x_hat_mean.cpu()
        alpha = alpha.cpu()
        beta = beta.cpu()

        num_rows = 4 if self.hparams.dataset == 'mnist' else 2
        fig, axes = plt.subplots(num_rows, num_images, figsize=(num_images, 4))
        for i in range(num_images):
            if self.hparams.dataset == 'mnist':
                axes[0, i].imshow(x[i].squeeze(), cmap='gray', vmin=0, vmax=1)
                axes[0, i].axis('off')
                axes[1, i].imshow(x_hat_mean[i].squeeze(), cmap='gray', vmin=0, vmax=1)
                axes[1, i].axis('off')
                axes[2, i].imshow(alpha[i].squeeze(), vmin=0, vmax=20)
                axes[2, i].set_title(f'{alpha[i].min().item():.2f} : {alpha[i].max().item():.2f}', fontsize=6, pad=-5)
                axes[2, i].axis('off')
                axes[3, i].imshow(beta[i].squeeze(), vmin=0, vmax=20)
                axes[3, i].axis('off')
                axes[3, i].set_title(f'{beta[i].min().item():.2f} : {beta[i].max().item():.2f}', fontsize=6, pad=-5)
                axes[3, i].axis('off')
                # add single combined colorbar for alpha/beta
                if i == 0:
                    fig.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=colors.Normalize(vmin=0, vmax=20)), ax=axes[2:4, :], orientation='horizontal')
            else:
                # cifar10
                img = np.transpose(x[i].numpy(), (1, 2, 0))
                img_hat = np.transpose(x_hat_mean[i].numpy(), (1, 2, 0))
                axes[0, i].imshow(img)
                axes[0, i].axis('off')
                axes[1, i].imshow(img_hat)
                axes[1, i].axis('off')
        return fig
    
    ## END: Common to Beta likelihood ##


    ## START: Specific to Gaussian Variational Posterior AND Beta likelihood ##

    def shared_step(self, batch, batch_idx, stage, stage_outputs):
        x, _ = batch

        likelihood_params, mu, log_var = self(x)
        
        alpha, beta = likelihood_params
        alpha = alpha.view(x.size(0), -1)
        beta = beta.view(x.size(0), -1)
        x = x.view(x.size(0), -1)

        log_prob_data = torch.distributions.Beta(alpha, beta).log_prob(x).sum(-1)
    
        kl = kl_standard_normal(mu, log_var).sum(-1)

        elbo = log_prob_data - kl
        neg_elbo = -elbo
        neg_elbo_mean = neg_elbo.mean()
        self.log(f'elbo_{stage}', -neg_elbo_mean, prog_bar=True)
        self.log(f'kl_{stage}', kl.mean())
        self.log(f'log_prob_data_{stage}', log_prob_data.mean())
         
        stage_outputs.append({'elbo': elbo.detach(), 'log_prob_data': log_prob_data.detach(), 'kl': kl.detach()})
        
        return neg_elbo_mean

    def shared_step_iwae(self, batch, batch_idx, stage, stage_outputs, k=200):
        x, _ = batch

        def reparameterize(mu, log_var, k):
            std = torch.exp(log_var / 2)
            q = torch.distributions.Normal(mu, std)
            z = q.rsample((k,))
            return z

        
        mu, log_var = self.encode(x)
        z = reparameterize(mu, log_var, k)
        # decode
        alpha, beta = self.decode(z.view(-1, z.shape[-1])) # self.decode(z) 
        alpha, beta = alpha.view(k, x.size(0), -1), beta.view(k, x.size(0), -1)

        # ELBO \approx \frac{1}{K} \sum_{k=1}^K ( log p(x | z_k) + log p(z_k) - log q(z_k | x) ), z_k \sim q(z|x)
        log_likelihood = torch.distributions.Beta(alpha, beta).log_prob(x).sum(-1)
        log_prior = torch.distributions.Normal(loc=0, scale=1).log_prob(z).sum(-1) # std normal prior
        log_var_post =  torch.distributions.Normal(loc=mu, scale=torch.exp(log_var / 2)).log_prob(z).sum(-1) # encoded var post

        # shape: (k, batch)
        log_importance_weights = log_likelihood + log_prior - log_var_post
        
        importance_weights = torch.logsumexp(log_importance_weights, 0)

        elbo_estimate = importance_weights.mean() # iwae elbo estimate for EACH image sample
        kl = kl_standard_normal(mu, log_var).sum(-1)

        stage_outputs.append({'elbo': importance_weights.detach(),
                              'log_prob_data': log_likelihood.detach(), 
                              'kl': kl.detach()
                              })

        return elbo_estimate
    
    ## END: Specific to Gaussian Variational Posterior AND Beta likelihood ##

class gauss_ks_VAE(L.LightningModule):
    def __init__(self, 
                 hidden_dim=500, 
                 latent_dim=20, 
                 keep_prob=0.9,
                 dataset='mnist',
                 learning_rate=1e-3
                 ):
        super(gauss_ks_VAE, self).__init__()

        assert dataset in ['mnist', 'cifar10'], f'unrecognized dataset: {dataset}'
        self.save_hyperparameters()

        self.n_output_encoder = 2 # gaussian
        self.n_output_decoder = 2 # ks
        if dataset == 'mnist':
            self.encoder = encoder_mnist(input_dim=int(28*28), hidden_dim=hidden_dim, latent_dim=latent_dim, keep_prob=keep_prob, n_output=self.n_output_encoder)
            self.decoder = decoder_mnist(input_dim=int(28*28), hidden_dim=hidden_dim, latent_dim=latent_dim, keep_prob=keep_prob, n_output=self.n_output_decoder)
        else:
            self.encoder = EncoderCIFAR(
                n_output=self.n_output_encoder, # for gaussian, KS, beta, etc
                latent_dim=latent_dim)
            self.decoder = DecoderCIFAR(
                n_output=self.n_output_decoder, 
                latent_dim=latent_dim)
        # Initialize lists to store outputs for logging
        # https://lightning.ai/releases/2.0.0/#bc-changes-pytorch
        self.train_outputs = []
        self.val_outputs = []
        self.test_outputs = []
    
    ## START: Common to ALL VAEs ##

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, 'train', self.train_outputs)
    
    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            fig = self.visualize_reconstructions(train_or_test='test', num_images=10)
            self.logger.experiment.log({"test reconstructions": wandb.Image(fig)})
            plt.close(fig)

            fig = self.visualize_reconstructions(train_or_test='train', num_images=10)
            self.logger.experiment.log({"train reconstructions": wandb.Image(fig)})
            plt.close(fig)
        return self.shared_step(batch, batch_idx, 'val', self.val_outputs)
    
    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, 'test', self.test_outputs)
    
    def shared_epoch_end(self, stage, stage_outputs):
        # Calculate average metrics
        avg_elbo = torch.stack([x['elbo'] for x in stage_outputs]).mean()
        avg_log_prob_data = torch.stack([x['log_prob_data'] for x in stage_outputs]).mean()
        avg_kl = torch.stack([x['kl'] for x in stage_outputs]).mean()
        
        # Log the average metrics
        self.log(f'avg_elbo_{stage}', avg_elbo)
        self.log(f'avg_log_prob_data_{stage}', avg_log_prob_data)
        self.log(f'avg_kl_{stage}', avg_kl)
        
        # Clear the outputs for the next epoch
        stage_outputs.clear()

    def on_train_epoch_end(self):
        self.shared_epoch_end('train', self.train_outputs)
    
    def on_validation_epoch_end(self):
        self.shared_epoch_end('val', self.val_outputs)

    def on_test_epoch_end(self):
        self.shared_epoch_end('test', self.test_outputs)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
    
    ## END: Common to ALL VAEs ##


    ## START: Common to all Gaussian Variational Posterior models ##
    
    def encode(self, x):
        x_encoded = self.encoder(x)
        mu, log_var = x_encoded.chunk(2, dim=-1)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return z
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    ## END: Common to all Gaussian Variational Posterior models ##


    ## START: Common to KS likelihood ##

    def decode(self, z):
        likelihood_params = self.decoder(z)
        log_a, log_b = likelihood_params.chunk(2, dim=-1)
        log_a, log_b = 5 * log_a, 10 * log_b # works without scaling, but scaling allows for more reasonable activations
        return (log_a, log_b)
        """
        # option 1: do nothing, works moderately well
        
        #log_a, log_b = log_a, 2 * log_b#torch.sigmoid(log_a), torch.sigmoid(log_b)
        
        # piecewise linear transformation \approx of sigmoid, without bad gradients
        # scale raw log_a and log_b so that neural activations stay more reasonable
        #log_a = piecewise_linear(2 * log_a, -2, 5, slope=0.1) #.01 # most reasonable bell shapes occur with log_a \in [0.1, 5]
        #log_b = piecewise_linear(5 * log_b, -2, 30, slope=0.1) # most reasonable bell shapes occur with log_b \in [0.1, 10]
        
        # MNIST
        if self.hparams.dataset == 'mnist':
            log_a, log_b = 2 * (log_a - 2), 7 * (log_b - 2)
        else:
            # CIFAR10
            log_a, log_b = 5 * log_a, 10 * log_b

        #log_a = -3 + (5 - (- 3)) * torch.sigmoid(log_a)
        #log_b = -3 + (30 - (-3)) * torch.sigmoid(log_b - 1)
        
        #log_a, log_b = .02 + torch.nn.functional.leaky_relu(log_a, negative_slope=.01), torch.nn.functional.leaky_relu(2 * log_b, negative_slope=.01)
        #log_a, log_b = 2 + 5 * torch.sigmoid(log_a), 7 * torch.sigmoid(log_b)
        #log_a, log_b = 6 * torch.sigmoid(log_a), 30 * torch.sigmoid(log_b)
        #log_a, log_b = 10 * log_a, 10 * log_b # kinda works, but blurry
        #log_a, log_b = 1e-6 + torch.nn.functional.softplus(log_a + 2), 1e-6 + 5 * torch.nn.functional.softplus(log_b + 2) # quickly produces NaNs
        return (log_a, log_b)
        """

    def visualize_reconstructions(self, train_or_test, num_images=10):
        # sample images
        self.eval()
        loader = self.trainer.datamodule.train_dataloader() if train_or_test == 'train' else self.trainer.val_dataloaders
        x, _ = next(iter(loader)) # first batch
        x = x.view(x.size(0), -1)
        x = x.to(self.device)
        with torch.no_grad():
            likelihood_params, _, _ = self(x)
        log_a, log_b = likelihood_params

        # reshape
        size = 28 if self.hparams.dataset == 'mnist' else 32
        channels = 1 if self.hparams.dataset == 'mnist' else 3
        log_a = log_a.view(-1, channels, size, size)
        log_b = log_b.view(-1, channels, size, size)
        x = x.view(-1, channels, size, size)

        # sample instead of closed form mean due to numerical instability
        x_hat_mean = KumaraswamyStable(log_a, log_b).sample((100,)).mean(0)

        x = x.to('cpu')
        x_hat_mean = x_hat_mean.to('cpu')
        log_a = log_a.cpu()
        log_b = log_b.cpu()

        lo, hi = -2.5, 7 # empirically find this is a reasonable range for log_a and log_b for visualization
        num_rows = 4 if self.hparams.dataset == 'mnist' else 2
        fig, axes = plt.subplots(num_rows, num_images, figsize=(num_images, 4))
        for i in range(num_images):
            if self.hparams.dataset == 'mnist':
                axes[0, i].imshow(x[i].squeeze(), cmap='gray', vmin=0, vmax=1)
                axes[0, i].axis('off')
                axes[1, i].imshow(x_hat_mean[i].squeeze(), cmap='gray', vmin=0, vmax=1)
                axes[1, i].axis('off')
                axes[2, i].imshow(log_a[i].squeeze(), vmin=lo, vmax=hi)
                axes[2, i].set_title(f'{log_a[i].min().item():.2f} : {log_a[i].max().item():.2f}', fontsize=6, pad=-5)
                axes[2, i].axis('off')
                axes[3, i].imshow(log_b[i].squeeze(), vmin=lo, vmax=hi)
                axes[3, i].set_title(f'{log_b[i].min().item():.2f} : {log_b[i].max().item():.2f}', fontsize=6, pad=-5)
                axes[3, i].axis('off')
                # add single combined colorbar for alpha/beta
                if i == 0:
                    fig.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=colors.Normalize(vmin=lo, vmax=hi)), ax=axes[2:4, :], orientation='horizontal')
            else:
                img = np.transpose(x[i].numpy(), (1, 2, 0))
                img_hat = np.transpose(x_hat_mean[i].numpy(), (1, 2, 0))
                axes[0, i].imshow(img)
                axes[0, i].axis('off')
                axes[1, i].imshow(img_hat)
                axes[1, i].axis('off')

        return fig

    ## END: Common to KS likelihood ##
    

    ## START: Specific to Gaussian Variational Posterior AND KS likelihood ##

    def shared_step(self, batch, batch_idx, stage, stage_outputs):
        x, _ = batch

        likelihood_params, mu, log_var = self(x)

        log_a, log_b = likelihood_params
        log_a = log_a.view(x.size(0), -1)
        log_b = log_b.view(x.size(0), -1)
        x = x.view(x.size(0), -1)

        log_prob_data = KumaraswamyStable(log_a, log_b).log_prob(x, max_grad_log_a_clamp=0.2).sum(-1) # log_a being driven to 0

        kl = kl_standard_normal(mu, log_var).sum(-1)

        elbo = log_prob_data - kl
        neg_elbo = -elbo
        neg_elbo_mean = neg_elbo.mean()
        self.log(f'elbo_{stage}', -neg_elbo_mean, prog_bar=True)
        self.log(f'kl_{stage}', kl.mean())
        self.log(f'log_prob_data_{stage}', log_prob_data.mean())
         
        stage_outputs.append({'elbo': elbo.detach(), 'log_prob_data': log_prob_data.detach(), 'kl': kl.detach()})
        
        return neg_elbo_mean
    
    def shared_step_iwae(self, batch, batch_idx, stage, stage_outputs, k=200):
        x, _ = batch

        def reparameterize(mu, log_var, k):
            std = torch.exp(log_var / 2)
            q = torch.distributions.Normal(mu, std)
            z = q.rsample((k,))
            return z

        
        mu, log_var = self.encode(x)
        z = reparameterize(mu, log_var, k)
        # decode
        log_a, log_b = self.decode(z.view(-1, z.shape[-1])) # self.decode(z) 
        log_a, log_b = log_a.view(k, x.size(0), -1), log_b.view(k, x.size(0), -1)

        # ELBO \approx \frac{1}{K} \sum_{k=1}^K ( log p(x | z_k) + log p(z_k) - log q(z_k | x) ), z_k \sim q(z|x)
        log_likelihood = KumaraswamyStable(log_a, log_b).log_prob(x, max_grad_log_a_clamp=0.2).sum(-1) # log_a being driven to 0
        log_prior = torch.distributions.Normal(loc=0, scale=1).log_prob(z).sum(-1) # std normal prior
        log_var_post =  torch.distributions.Normal(loc=mu, scale=torch.exp(log_var / 2)).log_prob(z).sum(-1) # encoded var post

        # shape: (k, batch)
        log_importance_weights = log_likelihood + log_prior - log_var_post
        
        importance_weights = torch.logsumexp(log_importance_weights, 0)

        elbo_estimate = importance_weights.mean() # iwae elbo estimate for EACH image sample
        kl = kl_standard_normal(mu, log_var).sum(-1)

        stage_outputs.append({'elbo': importance_weights.detach(),
                              'log_prob_data': log_likelihood.detach(), 
                              'kl': kl.detach()
                              })

        return elbo_estimate

    ## END: Specific to Gaussian Variational Posterior AND KS likelihood ##


#### Kumaraswamy Variational Posterior ####

def logging_ks_latent_space(self, stage, batch_idx, log_a, log_b, log_figure):
    self.log(f'log_a_mean_{stage}', log_a.mean())
    self.log(f'log_b_mean_{stage}', log_b.mean())
    self.log(f'log_a_median_{stage}', torch.median(log_a))
    self.log(f'log_b_median_{stage}', torch.median(log_b))
    self.log(f'log_a_min_{stage}', log_a.min())
    self.log(f'log_b_min_{stage}', log_b.min())
    self.log(f'log_a_max_{stage}', log_a.max())
    self.log(f'log_b_max_{stage}', log_b.max())

    if stage in ['val', 'test'] and batch_idx == 0 and log_figure:
        # for the first 10 samples in the batch, log a scatter plot log_a, log_b
        fig, axs = plt.subplots(1, 10, sharey=True)
        log_a_min = log_a.min().item()
        log_a_max = log_a.max().item()
        log_b_min = log_b.min().item()
        log_b_max = log_b.max().item()
        min_ = min(log_a_min, log_b_min)
        max_ = max(log_a_max, log_b_max)

        # round min to nearest integer below
        min_ = np.floor(min_)
        max_ = np.ceil(max_)

        for i in range(10):
            axs[i].scatter(log_a[i].cpu().detach().numpy(), log_b[i].cpu().detach().numpy())
            axs[i].set_xlim(min_, max_)
            axs[i].set_ylim(min_, max_)
            axs[i].set_aspect('equal')
            # draw horizontal line at y=0, vertical line at x=0
            if min_ < 0:
                axs[i].axhline(0, color='black', linewidth=0.2)
                axs[i].axvline(0, color='black', linewidth=0.2)
        
        plt.tight_layout()
        #plt.savefig('ks_latent_scatter.png')
        plt.close(fig)
        
        self.logger.experiment.log({f"ks_latent_{stage}_{batch_idx}": wandb.Image(fig)})

        return None
    
class ks_cb_VAE(L.LightningModule):
    def __init__(self, 
                 hidden_dim=500, 
                 latent_dim=20, 
                 keep_prob=0.9,
                 dataset='mnist',
                 learning_rate=1e-3,
                ):
        super(ks_cb_VAE, self).__init__()

        assert dataset in ['mnist', 'cifar10'], f'unrecognized dataset: {dataset}'
        self.save_hyperparameters()

        self.n_output_encoder = 2 # kumawaswamy
        self.n_output_decoder = 1 # continuous bernoulli
        if dataset == 'mnist':
            self.encoder = encoder_mnist(input_dim=int(28*28), hidden_dim=hidden_dim, latent_dim=latent_dim, keep_prob=keep_prob, n_output=self.n_output_encoder)
            self.decoder = decoder_mnist(input_dim=int(28*28), hidden_dim=hidden_dim, latent_dim=latent_dim, keep_prob=keep_prob, n_output=self.n_output_decoder)
        else:
            self.encoder = EncoderCIFAR(
                n_output=self.n_output_encoder, # for gaussian, KS, beta, etc
                latent_dim=latent_dim)
            self.decoder = DecoderCIFAR(n_output=self.n_output_decoder, latent_dim=latent_dim)
        # Initialize lists to store outputs for logging
        # https://lightning.ai/releases/2.0.0/#bc-changes-pytorch
        self.train_outputs = []
        self.val_outputs = []
        self.test_outputs = []
    
    ## START: Common to ALL VAEs ##   

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, 'train', self.train_outputs)
    
    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            fig = self.visualize_reconstructions(train_or_test='test', num_images=10)
            self.logger.experiment.log({"test reconstructions": wandb.Image(fig)})
            plt.close(fig)

            fig = self.visualize_reconstructions(train_or_test='train', num_images=10)
            self.logger.experiment.log({"train reconstructions": wandb.Image(fig)})
            plt.close(fig)
        return self.shared_step(batch, batch_idx, 'val', self.val_outputs)
    
    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, 'test', self.test_outputs)
    
    def shared_epoch_end(self, stage, stage_outputs):
        # Calculate average metrics
        avg_elbo = torch.stack([x['elbo'] for x in stage_outputs]).mean()
        avg_log_prob_data = torch.stack([x['log_prob_data'] for x in stage_outputs]).mean()
        avg_kl = torch.stack([x['kl'] for x in stage_outputs]).mean()
        
        # Log the average metrics
        self.log(f'avg_elbo_{stage}', avg_elbo)
        self.log(f'avg_log_prob_data_{stage}', avg_log_prob_data)
        self.log(f'avg_kl_{stage}', avg_kl)
        
        # Clear the outputs for the next epoch
        stage_outputs.clear()

    def on_train_epoch_end(self):
        self.shared_epoch_end('train', self.train_outputs)
    
    def on_validation_epoch_end(self):
        self.shared_epoch_end('val', self.val_outputs)

    def on_test_epoch_end(self):
        self.shared_epoch_end('test', self.test_outputs)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
    
    ## END: Common to ALL VAEs ##


    ## START: Common to all KS Variational Posterior models ##

    def encode(self, x):
        x_encoded = self.encoder(x)
        log_a, log_b = x_encoded.chunk(2, dim=-1)
        return log_a, log_b
    
    def reparameterize(self, log_a, log_b):
        q = KumaraswamyStable(log_a, log_b)
        z = q.rsample()
        return z
    
    def forward(self, x):
        log_a, log_b = self.encode(x)
        z = self.reparameterize(log_a, log_b)
        return self.decode(z), log_a, log_b
    
    ## END: Common to all KS Variational Posterior models ##


    ## START: Common to CB likelihood ##

    def decode(self, z):
        lambda_logit = self.decoder(z)
        return lambda_logit

    def visualize_reconstructions(self, train_or_test, num_images=10):
        # sample images
        self.eval()
        loader = self.trainer.datamodule.train_dataloader() if train_or_test == 'train' else self.trainer.val_dataloaders
        x, _ = next(iter(loader)) # first batch
        x = x.view(x.size(0), -1)
        x = x.to(self.device)
        with torch.no_grad():
            likelihood_params, _, _ = self(x)

        # reshape
        size = 28 if self.hparams.dataset == 'mnist' else 32
        channels = 1 if self.hparams.dataset == 'mnist' else 3
        likelihood_params = likelihood_params.view(-1, channels, size, size)
        x = x.view(-1, channels, size, size)

        x_hat_mean = torch.distributions.ContinuousBernoulli(logits=likelihood_params).mean

        x = x.cpu()
        x_hat_mean = x_hat_mean.cpu()

        fig, axes = plt.subplots(2, num_images, figsize=(num_images, 2))
        for i in range(num_images):
            if self.hparams.dataset == 'mnist':
                axes[0, i].imshow(x[i].squeeze(), cmap='gray', vmin=0, vmax=1)
                axes[0, i].axis('off')
                axes[1, i].imshow(x_hat_mean[i].squeeze(), cmap='gray', vmin=0, vmax=1)
                axes[1, i].axis('off')
            else:
                # cifar10
                img = np.transpose(x[i].numpy(), (1, 2, 0))
                img_hat = np.transpose(x_hat_mean[i].numpy(), (1, 2, 0))
                axes[0, i].imshow(img)
                axes[0, i].axis('off')
                axes[1, i].imshow(img_hat)
                axes[1, i].axis('off')
        return fig

    ## END: Common to CB likelihood ##
    
    
    ## START: Specific to KS Variational Posterior AND CB likelihood ##

    def shared_step(self, batch, batch_idx, stage, stage_outputs, log_figure=True):
        x, _ = batch

        likelihood_params, log_a, log_b = self(x)
        lambda_logit = likelihood_params

        lambda_logit = lambda_logit.view(x.size(0), -1)
        x = x.view(x.size(0), -1)

        log_prob_data = torch.distributions.ContinuousBernoulli(logits=lambda_logit).log_prob(x).sum(-1)
        
        # kl between KS(log_a, log_b) and U(0, 1). Using torch implementation, no instability.
        kl = - torch.distributions.Kumaraswamy(torch.exp(log_a), torch.exp(log_b)).entropy().sum(-1) 

        elbo = log_prob_data - kl
        neg_elbo = -elbo
        neg_elbo_mean = neg_elbo.mean()
        self.log(f'elbo_{stage}', -neg_elbo_mean, prog_bar=True)
        self.log(f'kl_{stage}', kl.mean())
        self.log(f'log_prob_data_{stage}', log_prob_data.mean())

        stage_outputs.append({'elbo': elbo.detach(), 'log_prob_data': log_prob_data.detach(), 'kl': kl.detach()})

        with torch.no_grad():
            logging_ks_latent_space(self, stage, batch_idx, log_a, log_b, log_figure)
   
        return neg_elbo_mean

    def shared_step_iwae(self, batch, batch_idx, stage, stage_outputs, log_figure=True, k=200):
        x, _ = batch

        log_a, log_b  = self.encode(x)
        z = KumaraswamyStable(log_a, log_b).rsample((k,))
        # decode
        lambda_logit = self.decode(z.view(-1, z.shape[-1])) # self.decode(z) 
        lambda_logit = lambda_logit.view(k, x.size(0), -1)

        # ELBO \approx \frac{1}{K} \sum_{k=1}^K ( log p(x | z_k) + log p(z_k) - log q(z_k | x) ), z_k \sim q(z|x)
        log_likelihood = torch.distributions.ContinuousBernoulli(logits=lambda_logit).log_prob(x).sum(-1)
        log_prior = torch.distributions.Uniform(low=0, high=1).log_prob(z).sum(-1) # uniform normal prior over U(0, 1)^d
        log_var_post = KumaraswamyStable(log_a, log_b).log_prob(z).sum(-1) # encoded var post

        # shape: (k, batch)
        log_importance_weights = log_likelihood + log_prior - log_var_post
        
        importance_weights = torch.logsumexp(log_importance_weights, 0)

        elbo_estimate = importance_weights.mean() # iwae elbo estimate for EACH image sample
        kl = - torch.distributions.Kumaraswamy(torch.exp(log_a), torch.exp(log_b)).entropy().sum(-1) 
        
        stage_outputs.append({'elbo': importance_weights.detach(),
                              'log_prob_data': log_likelihood.detach(), 
                              'kl': kl.detach()
                              })

        return elbo_estimate

    ## END: Specific to KS Variational Posterior AND CB likelihood ## 

class ks_beta_VAE(L.LightningModule):
    def __init__(self, 
                 hidden_dim=500, 
                 latent_dim=20, 
                 keep_prob=0.9,
                 dataset='mnist',
                 learning_rate=1e-3
                 ):
        super(ks_beta_VAE, self).__init__()

        assert dataset in ['mnist', 'cifar10'], f'unrecognized dataset: {dataset}'
        self.save_hyperparameters()

        self.n_output_encoder = 2 # kumawaswamy
        self.n_output_decoder = 2 # beta
        if dataset == 'mnist':
            self.encoder = encoder_mnist(input_dim=int(28*28), hidden_dim=hidden_dim, latent_dim=latent_dim, keep_prob=keep_prob, n_output=self.n_output_encoder)
            self.decoder = decoder_mnist(input_dim=int(28*28), hidden_dim=hidden_dim, latent_dim=latent_dim, keep_prob=keep_prob, n_output=self.n_output_decoder)
        else:
            self.encoder = EncoderCIFAR(
                n_output=self.n_output_encoder, # for gaussian, KS, beta, etc
                latent_dim=latent_dim)
            self.decoder = DecoderCIFAR(n_output=self.n_output_decoder, latent_dim=latent_dim)
        # Initialize lists to store outputs for logging
        # https://lightning.ai/releases/2.0.0/#bc-changes-pytorch
        self.train_outputs = []
        self.val_outputs = []
        self.test_outputs = []

    ## START: Common to ALL VAEs ##   

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, 'train', self.train_outputs)
    
    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            fig = self.visualize_reconstructions(train_or_test='test', num_images=10)
            self.logger.experiment.log({"test reconstructions": wandb.Image(fig)})
            plt.close(fig)

            fig = self.visualize_reconstructions(train_or_test='train', num_images=10)
            self.logger.experiment.log({"train reconstructions": wandb.Image(fig)})
            plt.close(fig)
        return self.shared_step(batch, batch_idx, 'val', self.val_outputs)
    
    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, 'test', self.test_outputs)
    
    def shared_epoch_end(self, stage, stage_outputs):
        # Calculate average metrics
        avg_elbo = torch.stack([x['elbo'] for x in stage_outputs]).mean()
        avg_log_prob_data = torch.stack([x['log_prob_data'] for x in stage_outputs]).mean()
        avg_kl = torch.stack([x['kl'] for x in stage_outputs]).mean()
        
        # Log the average metrics
        self.log(f'avg_elbo_{stage}', avg_elbo)
        self.log(f'avg_log_prob_data_{stage}', avg_log_prob_data)
        self.log(f'avg_kl_{stage}', avg_kl)
        
        # Clear the outputs for the next epoch
        stage_outputs.clear()

    def on_train_epoch_end(self):
        self.shared_epoch_end('train', self.train_outputs)
    
    def on_validation_epoch_end(self):
        self.shared_epoch_end('val', self.val_outputs)

    def on_test_epoch_end(self):
        self.shared_epoch_end('test', self.test_outputs)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
    
    ## END: Common to ALL VAEs ##


    ## START: Common to all KS Variational Posterior models ##

    def encode(self, x):
        x_encoded = self.encoder(x)
        log_a, log_b = x_encoded.chunk(2, dim=-1)
        return log_a, log_b
    
    def reparameterize(self, log_a, log_b):
        q = KumaraswamyStable(log_a, log_b)
        z = q.rsample()
        return z
    
    def forward(self, x):
        log_a, log_b = self.encode(x)
        z = self.reparameterize(log_a, log_b)
        return self.decode(z), log_a, log_b
    
    ## END: Common to all KS Variational Posterior models ##


    ## START: Common to Beta likelihood ##

    def decode(self, z):
        likelihood_params = self.decoder(z)
        log_alpha, log_beta = likelihood_params.chunk(2, dim=-1)
        alpha, beta = 1e-6 + torch.nn.functional.softplus(log_alpha), 1e-6 + torch.nn.functional.softplus(log_beta)
        return (alpha, beta)

    def visualize_reconstructions(self, train_or_test, num_images=10):
        # sample images
        self.eval()
        loader = self.trainer.datamodule.train_dataloader() if train_or_test == 'train' else self.trainer.val_dataloaders
        x, _ = next(iter(loader)) # first batch
        x = x.view(x.size(0), -1)
        x = x.to(self.device)
        with torch.no_grad():
            likelihood_params, _, _ = self(x)
        alpha, beta = likelihood_params

        # reshape
        size = 28 if self.hparams.dataset == 'mnist' else 32
        channels = 1 if self.hparams.dataset == 'mnist' else 3
        alpha = alpha.view(-1, channels, size, size)
        beta = beta.view(-1, channels, size, size)
        x = x.view(-1, channels, size, size)

        x_hat_mean = torch.distributions.Beta(alpha, beta).mean

        x = x.cpu()
        x_hat_mean = x_hat_mean.cpu()
        alpha = alpha.cpu()
        beta = beta.cpu()

        num_rows = 4 if self.hparams.dataset == 'mnist' else 2
        fig, axes = plt.subplots(num_rows, num_images, figsize=(num_images, 4))
        for i in range(num_images):
            if self.hparams.dataset == 'mnist':
                axes[0, i].imshow(x[i].squeeze(), cmap='gray', vmin=0, vmax=1)
                axes[0, i].axis('off')
                axes[1, i].imshow(x_hat_mean[i].squeeze(), cmap='gray', vmin=0, vmax=1)
                axes[1, i].axis('off')
                axes[2, i].imshow(alpha[i].squeeze(), vmin=0, vmax=20)
                axes[2, i].set_title(f'{alpha[i].min().item():.2f} : {alpha[i].max().item():.2f}', fontsize=6, pad=-5)
                axes[2, i].axis('off')
                axes[3, i].imshow(beta[i].squeeze(), vmin=0, vmax=20)
                axes[3, i].axis('off')
                axes[3, i].set_title(f'{beta[i].min().item():.2f} : {beta[i].max().item():.2f}', fontsize=6, pad=-5)
                axes[3, i].axis('off')
                # add single combined colorbar for alpha/beta
                if i == 0:
                    fig.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=colors.Normalize(vmin=0, vmax=20)), ax=axes[2:4, :], orientation='horizontal')
            else:
                # cifar10
                img = np.transpose(x[i].numpy(), (1, 2, 0))
                img_hat = np.transpose(x_hat_mean[i].numpy(), (1, 2, 0))
                axes[0, i].imshow(img)
                axes[0, i].axis('off')
                axes[1, i].imshow(img_hat)
                axes[1, i].axis('off')
        return fig
    
    ## END: Common to Beta likelihood ##
    
    
    ## START: Specific to KS Variational Posterior AND Beta likelihood ##

    def shared_step(self, batch, batch_idx, stage, stage_outputs, log_figure=True):
        x, _ = batch

        likelihood_params, log_a, log_b = self(x)
        alpha, beta = likelihood_params

        alpha = alpha.view(x.size(0), -1)
        beta = beta.view(x.size(0), -1)
        x = x.view(x.size(0), -1)
        
        log_prob_data = torch.distributions.Beta(alpha, beta).log_prob(x).sum(-1)
        
        kl = - torch.distributions.Kumaraswamy(torch.exp(log_a), torch.exp(log_b)).entropy().sum(-1)

        elbo = log_prob_data - kl
        neg_elbo = -elbo
        neg_elbo_mean = neg_elbo.mean()
        self.log(f'elbo_{stage}', -neg_elbo_mean, prog_bar=True)
        self.log(f'kl_{stage}', kl.mean())
        self.log(f'log_prob_data_{stage}', log_prob_data.mean())

        stage_outputs.append({'elbo': elbo.detach(), 'log_prob_data': log_prob_data.detach(), 'kl': kl.detach()})

        with torch.no_grad():
            logging_ks_latent_space(self, stage, batch_idx, log_a, log_b, log_figure)

        return neg_elbo_mean
    
    def shared_step_iwae(self, batch, batch_idx, stage, stage_outputs, log_figure=True, k=200):
        x, _ = batch

        log_a, log_b  = self.encode(x)
        z = KumaraswamyStable(log_a, log_b).rsample((k,))
        # decode
        alpha, beta = self.decode(z.view(-1, z.shape[-1])) # self.decode(z) 
        alpha, beta = alpha.view(k, x.size(0), -1), beta.view(k, x.size(0), -1)

        # ELBO \approx \frac{1}{K} \sum_{k=1}^K ( log p(x | z_k) + log p(z_k) - log q(z_k | x) ), z_k \sim q(z|x)
        log_likelihood = torch.distributions.Beta(alpha, beta).log_prob(x).sum(-1)
        log_prior = torch.distributions.Uniform(low=0, high=1).log_prob(z).sum(-1) # uniform normal prior over U(0, 1)^d
        log_var_post = KumaraswamyStable(log_a, log_b).log_prob(z).sum(-1) # encoded var post

        # shape: (k, batch)
        log_importance_weights = log_likelihood + log_prior - log_var_post
        
        importance_weights = torch.logsumexp(log_importance_weights, 0)

        elbo_estimate = importance_weights.mean() # iwae elbo estimate for EACH image sample
        kl = - torch.distributions.Kumaraswamy(torch.exp(log_a), torch.exp(log_b)).entropy().sum(-1) 
        
        stage_outputs.append({'elbo': importance_weights.detach(),
                              'log_prob_data': log_likelihood.detach(), 
                              'kl': kl.detach()
                              })

        return elbo_estimate

    ## END: Specific to Gaussian Variational Posterior AND CB likelihood ## 
    
class ks_ks_VAE(L.LightningModule):
    def __init__(self, 
                 hidden_dim=500, 
                 latent_dim=20, 
                 keep_prob=0.9,
                 dataset='mnist',
                 learning_rate=1e-3,
                ):
        super(ks_ks_VAE, self).__init__()

        assert dataset in ['mnist', 'cifar10'], f'unrecognized dataset: {dataset}'
        self.save_hyperparameters()

        self.n_output_encoder = 2 # ks
        self.n_output_decoder = 2 # ks
        if dataset == 'mnist':
            self.encoder = encoder_mnist(input_dim=int(28*28), hidden_dim=hidden_dim, latent_dim=latent_dim, keep_prob=keep_prob, n_output=self.n_output_encoder)
            self.decoder = decoder_mnist(input_dim=int(28*28), hidden_dim=hidden_dim, latent_dim=latent_dim, keep_prob=keep_prob, n_output=self.n_output_decoder)
        else:
            self.encoder = EncoderCIFAR(
                n_output=self.n_output_encoder, # for gaussian, KS, beta, etc
                latent_dim=latent_dim)
            self.decoder = DecoderCIFAR(
                n_output=self.n_output_decoder, 
                latent_dim=latent_dim)
        # Initialize lists to store outputs for logging
        # https://lightning.ai/releases/2.0.0/#bc-changes-pytorch
        self.train_outputs = []
        self.val_outputs = []
        self.test_outputs = []
    
    ## START: Common to ALL VAEs ##   

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, 'train', self.train_outputs)
    
    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            fig = self.visualize_reconstructions(train_or_test='test', num_images=10)
            self.logger.experiment.log({"test reconstructions": wandb.Image(fig)})
            plt.close(fig)

            fig = self.visualize_reconstructions(train_or_test='train', num_images=10)
            self.logger.experiment.log({"train reconstructions": wandb.Image(fig)})
            plt.close(fig)
        return self.shared_step(batch, batch_idx, 'val', self.val_outputs)
    
    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, 'test', self.test_outputs)
    
    def shared_epoch_end(self, stage, stage_outputs):
        # Calculate average metrics
        avg_elbo = torch.stack([x['elbo'] for x in stage_outputs]).mean()
        avg_log_prob_data = torch.stack([x['log_prob_data'] for x in stage_outputs]).mean()
        avg_kl = torch.stack([x['kl'] for x in stage_outputs]).mean()
        
        # Log the average metrics
        self.log(f'avg_elbo_{stage}', avg_elbo)
        self.log(f'avg_log_prob_data_{stage}', avg_log_prob_data)
        self.log(f'avg_kl_{stage}', avg_kl)
        
        # Clear the outputs for the next epoch
        stage_outputs.clear()

    def on_train_epoch_end(self):
        self.shared_epoch_end('train', self.train_outputs)
    
    def on_validation_epoch_end(self):
        self.shared_epoch_end('val', self.val_outputs)

    def on_test_epoch_end(self):
        self.shared_epoch_end('test', self.test_outputs)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
    
    ## END: Common to ALL VAEs ##


    ## START: Common to all KS Variational Posterior models ##

    def encode(self, x):
        x_encoded = self.encoder(x)
        log_a, log_b = x_encoded.chunk(2, dim=-1)
        return log_a, log_b
    
    def reparameterize(self, log_a, log_b):
        q = KumaraswamyStable(log_a, log_b)
        z = q.rsample()
        return z
    
    def forward(self, x):
        log_a, log_b = self.encode(x)
        z = self.reparameterize(log_a, log_b)
        return self.decode(z), log_a, log_b
    
    ## END: Common to all KS Variational Posterior models ##
    
    
    ## START: Common to KS likelihood ##

    def decode(self, z):
        likelihood_params = self.decoder(z)
        log_a, log_b = likelihood_params.chunk(2, dim=-1)
        log_a, log_b = 5 * log_a, 10 * log_b # works without scaling, but scaling allows for more reasonable activations
        return (log_a, log_b)

    def visualize_reconstructions(self, train_or_test, num_images=10):
        # sample images
        self.eval()
        loader = self.trainer.datamodule.train_dataloader() if train_or_test == 'train' else self.trainer.val_dataloaders
        x, _ = next(iter(loader)) # first batch
        x = x.view(x.size(0), -1)
        x = x.to(self.device)
        with torch.no_grad():
            likelihood_params, _, _ = self(x)
        log_a, log_b = likelihood_params

        # reshape
        size = 28 if self.hparams.dataset == 'mnist' else 32
        channels = 1 if self.hparams.dataset == 'mnist' else 3
        log_a = log_a.view(-1, channels, size, size)
        log_b = log_b.view(-1, channels, size, size)
        x = x.view(-1, channels, size, size)

        # sample instead of closed form mean due to numerical instability
        x_hat_mean = KumaraswamyStable(log_a, log_b).sample((100,)).mean(0) # out of memory at 500 samples with bs 500

        x = x.to('cpu')
        x_hat_mean = x_hat_mean.to('cpu')
        log_a = log_a.cpu()
        log_b = log_b.cpu()

        lo, hi = -2.5, 7 # empirically find this is a reasonable range for log_a and log_b for visualization
        num_rows = 4 if self.hparams.dataset == 'mnist' else 2
        fig, axes = plt.subplots(num_rows, num_images, figsize=(num_images, 4))
        for i in range(num_images):
            if self.hparams.dataset == 'mnist':
                axes[0, i].imshow(x[i].squeeze(), cmap='gray', vmin=0, vmax=1)
                axes[0, i].axis('off')
                axes[1, i].imshow(x_hat_mean[i].squeeze(), cmap='gray', vmin=0, vmax=1)
                axes[1, i].axis('off')
                axes[2, i].imshow(log_a[i].squeeze(), vmin=lo, vmax=hi)
                axes[2, i].set_title(f'{log_a[i].min().item():.2f} : {log_a[i].max().item():.2f}', fontsize=6, pad=-5)
                axes[2, i].axis('off')
                axes[3, i].imshow(log_b[i].squeeze(), vmin=lo, vmax=hi)
                axes[3, i].set_title(f'{log_b[i].min().item():.2f} : {log_b[i].max().item():.2f}', fontsize=6, pad=-5)
                axes[3, i].axis('off')
                # add single combined colorbar for alpha/beta
                if i == 0:
                    fig.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=colors.Normalize(vmin=lo, vmax=hi)), ax=axes[2:4, :], orientation='horizontal')
            else:
                img = np.transpose(x[i].numpy(), (1, 2, 0))
                img_hat = np.transpose(x_hat_mean[i].numpy(), (1, 2, 0))
                axes[0, i].imshow(img)
                axes[0, i].axis('off')
                axes[1, i].imshow(img_hat)
                axes[1, i].axis('off')

        return fig

    ## END: Common to KS likelihood ##


    ## START: Specific to KS Variational Posterior AND KS likelihood ##

    def shared_step(self, batch, batch_idx, stage, stage_outputs, log_figure=True):
        x, _ = batch

        likelihood_params, log_a_latent, log_b_latent = self(x)
        log_a_likelihood, log_b_likelihood = likelihood_params
        log_a_likelihood = log_a_likelihood.view(x.size(0), -1)
        log_b_likelihood = log_b_likelihood.view(x.size(0), -1)
        x = x.view(x.size(0), -1)
    
        log_prob_data = KumaraswamyStable(log_a_likelihood, log_b_likelihood).log_prob(x, max_grad_log_a_clamp=0.2).sum(-1)
        
        kl = - torch.distributions.Kumaraswamy(torch.exp(log_a_latent), torch.exp(log_b_latent)).entropy().sum(-1)

        """
        # for debugging high kl samples
        if batch_idx == 24:
            print('stop')
        # inspect the log_a/ log_b with the highest KL
        # 1. find the input with the highest summed (over latent dims) KL
        kl_max_input_indx = kl.argmax()
        # 2. sort the latent dimensions by KL
        non_red_kl = - torch.distributions.Kumaraswamy(torch.exp(log_a_latent), torch.exp(log_b_latent)).entropy()
        kl_max_dim_indices = non_red_kl[kl_max_input_indx].sort()[1]
        # 3. find the values of log_prob, kl, log_a_latent, log_b_latent sorted by KL
        log_prob_data_of_max_kl = log_prob_data[kl_max_input_indx]
        kl_sorted = non_red_kl[kl_max_input_indx][kl_max_dim_indices][:5]
        log_a_latent_sorted = log_a_latent[kl_max_input_indx][kl_max_dim_indices][:5]
        log_b_latent_sorted = log_b_latent[kl_max_input_indx][kl_max_dim_indices][:5]
        # 4. print the values torch tensors with pretty formatting.
        # only print first decimal place
        torch.set_printoptions(precision=1)
        print(f"\tmax kl {kl[kl_max_input_indx]}")
        print(f"\tlog prob of max kl {log_prob_data_of_max_kl}")
        print(f"\tper dim latent kl of max kl {kl_sorted}")
        print(f"\tlog_a {log_a_latent_sorted}")
        print(f"\tlog_b {log_b_latent_sorted}")
        # reset print options
        torch.set_printoptions(precision=8)
        """

        elbo = log_prob_data - kl
        neg_elbo = -elbo
        neg_elbo_mean = neg_elbo.mean()
        self.log(f'elbo_{stage}', -neg_elbo_mean, prog_bar=True)
        self.log(f'kl_{stage}', kl.mean())
        self.log(f'log_prob_data_{stage}', log_prob_data.mean())

        stage_outputs.append({'elbo': elbo.detach(), 'log_prob_data': log_prob_data.detach(), 'kl': kl.detach()})

        with torch.no_grad():
            logging_ks_latent_space(self, stage, batch_idx, log_a_latent, log_b_latent, log_figure)

        return neg_elbo_mean
    
    def shared_step_iwae(self, batch, batch_idx, stage, stage_outputs, log_figure=True, k=200):
        x, _ = batch

        log_a, log_b  = self.encode(x)
        z = KumaraswamyStable(log_a, log_b).rsample((k,))
        # decode
        log_a_likelihood, log_b_likelihood = self.decode(z.view(-1, z.shape[-1])) # self.decode(z) 
        log_a_likelihood, log_b_likelihood = log_a_likelihood.view(k, x.size(0), -1), log_b_likelihood.view(k, x.size(0), -1)

        # ELBO \approx \frac{1}{K} \sum_{k=1}^K ( log p(x | z_k) + log p(z_k) - log q(z_k | x) ), z_k \sim q(z|x)
        log_likelihood = KumaraswamyStable(log_a_likelihood, log_b_likelihood).log_prob(x, max_grad_log_a_clamp=0.2).sum(-1)
        log_prior = torch.distributions.Uniform(low=0, high=1).log_prob(z).sum(-1) # uniform normal prior over U(0, 1)^d
        log_var_post = KumaraswamyStable(log_a, log_b).log_prob(z).sum(-1) # encoded var post

        # shape: (k, batch)
        log_importance_weights = log_likelihood + log_prior - log_var_post
        
        importance_weights = torch.logsumexp(log_importance_weights, 0)

        elbo_estimate = importance_weights.mean() # iwae elbo estimate for EACH image sample
        kl = - torch.distributions.Kumaraswamy(torch.exp(log_a), torch.exp(log_b)).entropy().sum(-1) 
        
        stage_outputs.append({'elbo': importance_weights.detach(),
                              'log_prob_data': log_likelihood.detach(), 
                              'kl': kl.detach()
                              })

        return elbo_estimate

    ## END: Specific to KS Variational Posterior AND KS likelihood ##


#### Beta Variational Posterior ####

def logging_beta_latent_space(self, stage, batch_idx, log_alpha, log_beta, log_figure):
    self.log(f'log_alpha_mean_{stage}', log_alpha.mean())
    self.log(f'log_beta_mean_{stage}', log_beta.mean())
    self.log(f'log_alpha_median_{stage}', torch.median(log_alpha))
    self.log(f'log_beta_median_{stage}', torch.median(log_beta))
    self.log(f'log_alpha_min_{stage}', log_alpha.min())
    self.log(f'log_beta_min_{stage}', log_beta.min())
    self.log(f'log_alpha_max_{stage}', log_alpha.max())
    self.log(f'log_beta_max_{stage}', log_beta.max())

    if stage in ['val', 'test'] and batch_idx == 0 and log_figure:
        # for the first 10 samples in the batch, log a scatter plot log_alpha, log_beta
        fig, axs = plt.subplots(1, 10, sharey=True)
        log_alpha_min = log_alpha.min().item()
        log_alpha_max = log_alpha.max().item()
        log_beta_min = log_beta.min().item()
        log_beta_max = log_beta.max().item()
        min_ = min(log_alpha_min, log_beta_min)
        max_ = max(log_alpha_max, log_beta_max)
        # ensure min and max are finite
        min_ = min(min_, -30)
        max_ = max(max_, 30)
        # round min to nearest integer below
        min_ = np.floor(min_)
        max_ = np.ceil(max_)

        for i in range(10):
            axs[i].scatter(log_alpha[i].cpu().detach().numpy(), log_beta[i].cpu().detach().numpy())
            axs[i].set_xlim(min_, max_)
            axs[i].set_ylim(min_, max_)
            axs[i].set_aspect('equal')
            # draw horizontal line at y=0, vertical line at x=0
            if min_ < 0:
                axs[i].axhline(0, color='black', linewidth=0.2)
                axs[i].axvline(0, color='black', linewidth=0.2)
        
        plt.tight_layout()
        plt.close(fig)
        
        if self.logger is not None:
            self.logger.experiment.log({f"beta_latent_{stage}_{batch_idx}": wandb.Image(fig)})

    return None
class beta_cb_VAE(L.LightningModule):
    def __init__(self, 
                 hidden_dim=500, 
                 latent_dim=20,
                 var_link_func='softplus',
                 keep_prob=0.9,
                 dataset='mnist',
                 learning_rate=1e-3,
                ):
        super(beta_cb_VAE, self).__init__()

        assert dataset in ['mnist', 'cifar10'], f'unrecognized dataset: {dataset}'
        self.save_hyperparameters()

        self.n_output_encoder = 2 # beta 
        self.n_output_decoder = 1 # continuous bernoulli
        if dataset == 'mnist':
            self.encoder = encoder_mnist(input_dim=int(28*28), hidden_dim=hidden_dim, latent_dim=latent_dim, keep_prob=keep_prob, n_output=self.n_output_encoder)
            self.decoder = decoder_mnist(input_dim=int(28*28), hidden_dim=hidden_dim, latent_dim=latent_dim, keep_prob=keep_prob, n_output=self.n_output_decoder)
        else:
            self.encoder = EncoderCIFAR(
                n_output=self.n_output_encoder, # for gaussian, KS, beta, etc
                latent_dim=latent_dim)
            self.decoder = DecoderCIFAR(n_output=self.n_output_decoder, latent_dim=latent_dim)
        # Initialize lists to store outputs for logging
        # https://lightning.ai/releases/2.0.0/#bc-changes-pytorch
        self.train_outputs = []
        self.val_outputs = []
        self.test_outputs = []
    
    ## START: Common to ALL VAEs ##   

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, 'train', self.train_outputs)
    
    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            fig = self.visualize_reconstructions(train_or_test='test', num_images=10)
            self.logger.experiment.log({"test reconstructions": wandb.Image(fig)})
            plt.close(fig)

            fig = self.visualize_reconstructions(train_or_test='train', num_images=10)
            self.logger.experiment.log({"train reconstructions": wandb.Image(fig)})
            plt.close(fig)
        return self.shared_step(batch, batch_idx, 'val', self.val_outputs)
    
    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, 'test', self.test_outputs)
    
    def shared_epoch_end(self, stage, stage_outputs):
        # Calculate average metrics
        avg_elbo = torch.stack([x['elbo'] for x in stage_outputs]).mean()
        avg_log_prob_data = torch.stack([x['log_prob_data'] for x in stage_outputs]).mean()
        avg_kl = torch.stack([x['kl'] for x in stage_outputs]).mean()
        
        # Log the average metrics
        self.log(f'avg_elbo_{stage}', avg_elbo)
        self.log(f'avg_log_prob_data_{stage}', avg_log_prob_data)
        self.log(f'avg_kl_{stage}', avg_kl)
        
        # Clear the outputs for the next epoch
        stage_outputs.clear()

    def on_train_epoch_end(self):
        self.shared_epoch_end('train', self.train_outputs)
    
    def on_validation_epoch_end(self):
        self.shared_epoch_end('val', self.val_outputs)

    def on_test_epoch_end(self):
        self.shared_epoch_end('test', self.test_outputs)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
    
    ## END: Common to ALL VAEs ##


    ## START: Common to all Beta Variational Posterior models ##

    def encode(self, x):
        x_encoded = self.encoder(x)
        log_alpha, log_beta = x_encoded.chunk(2, dim=-1)
        if self.hparams.var_link_func == 'softplus':
            alpha, beta = 1e-6 + torch.nn.functional.softplus(log_alpha), 1e-6 + torch.nn.functional.softplus(log_beta)
        elif self.hparams.var_link_func == 'exp':
            alpha, beta = torch.exp(log_alpha), torch.exp(log_beta)
        return alpha, beta
    
    def reparameterize(self, alpha, beta):
        q = torch.distributions.Beta(alpha, beta)
        z = q.rsample()
        return z
    
    def forward(self, x):
        alpha, beta = self.encode(x)
        z = self.reparameterize(alpha, beta)
        return self.decode(z), alpha, beta
    
    ## END: Common to all Beta Variational Posterior models ##


    ## START: Common to CB likelihood ##

    def decode(self, z):
        lambda_logit = self.decoder(z)
        return lambda_logit

    def visualize_reconstructions(self, train_or_test, num_images=10):
        # sample images
        self.eval()
        loader = self.trainer.datamodule.train_dataloader() if train_or_test == 'train' else self.trainer.val_dataloaders
        x, _ = next(iter(loader)) # first batch
        x = x.view(x.size(0), -1)
        x = x.to(self.device)
        with torch.no_grad():
            likelihood_params, _, _ = self(x)

        # reshape
        size = 28 if self.hparams.dataset == 'mnist' else 32
        channels = 1 if self.hparams.dataset == 'mnist' else 3
        likelihood_params = likelihood_params.view(-1, channels, size, size)
        x = x.view(-1, channels, size, size)

        x_hat_mean = torch.distributions.ContinuousBernoulli(logits=likelihood_params).mean

        x = x.cpu()
        x_hat_mean = x_hat_mean.cpu()

        fig, axes = plt.subplots(2, num_images, figsize=(num_images, 2))
        for i in range(num_images):
            if self.hparams.dataset == 'mnist':
                axes[0, i].imshow(x[i].squeeze(), cmap='gray', vmin=0, vmax=1)
                axes[0, i].axis('off')
                axes[1, i].imshow(x_hat_mean[i].squeeze(), cmap='gray', vmin=0, vmax=1)
                axes[1, i].axis('off')
            else:
                # cifar10
                img = np.transpose(x[i].numpy(), (1, 2, 0))
                img_hat = np.transpose(x_hat_mean[i].numpy(), (1, 2, 0))
                axes[0, i].imshow(img)
                axes[0, i].axis('off')
                axes[1, i].imshow(img_hat)
                axes[1, i].axis('off')
        return fig

    ## END: Common to CB likelihood ##
    
    
    ## START: Specific to Beta Variational Posterior AND CB likelihood ##

    def shared_step(self, batch, batch_idx, stage, stage_outputs, log_figure=True):
        x, _ = batch

        likelihood_params, alpha, beta = self(x)
        lambda_logit = likelihood_params

        lambda_logit = lambda_logit.view(x.size(0), -1)
        x = x.view(x.size(0), -1)

        log_prob_data = torch.distributions.ContinuousBernoulli(logits=lambda_logit).log_prob(x).sum(-1)
        
        # kl between Beta(alpha, beta) and U(0, 1). Using torch implementation, no instability.
        kl = - torch.distributions.Beta(alpha, beta).entropy().sum(-1) 

        elbo = log_prob_data - kl
        neg_elbo = -elbo
        neg_elbo_mean = neg_elbo.mean()
        self.log(f'elbo_{stage}', -neg_elbo_mean, prog_bar=True)
        self.log(f'kl_{stage}', kl.mean())
        self.log(f'log_prob_data_{stage}', log_prob_data.mean())

        stage_outputs.append({'elbo': elbo.detach(), 'log_prob_data': log_prob_data.detach(), 'kl': kl.detach()})

        with torch.no_grad():
            log_alpha, log_beta = torch.log(1e-16 + alpha), torch.log(1e-16 + beta)
            logging_beta_latent_space(self, stage, batch_idx, log_alpha, log_beta, log_figure)
   
        return neg_elbo_mean
    
    def shared_step_iwae(self, batch, batch_idx, stage, stage_outputs, log_figure=True, k=200):
        x, _ = batch

        alpha, beta  = self.encode(x)
        z = torch.distributions.Beta(alpha, beta).rsample((k,))
        # decode
        lambda_logit = self.decode(z.view(-1, z.shape[-1])) # self.decode(z)
        lambda_logit = lambda_logit.view(k, x.size(0), -1)


        # ELBO \approx \frac{1}{K} \sum_{k=1}^K ( log p(x | z_k) + log p(z_k) - log q(z_k | x) ), z_k \sim q(z|x)
        log_likelihood = torch.distributions.ContinuousBernoulli(logits=lambda_logit).log_prob(x).sum(-1)
        log_prior = torch.distributions.Uniform(low=0, high=1).log_prob(z).sum(-1) # uniform normal prior over U(0, 1)^d
        log_var_post = torch.distributions.Beta(alpha, beta).log_prob(z).sum(-1) # encoded var post

        # shape: (k, batch)
        log_importance_weights = log_likelihood + log_prior - log_var_post
        
        importance_weights = torch.logsumexp(log_importance_weights, 0)

        elbo_estimate = importance_weights.mean() # iwae elbo estimate for EACH image sample
        
        # kl between Beta(alpha, beta) and U(0, 1). Using torch implementation, no instability.
        kl = - torch.distributions.Beta(alpha, beta).entropy().sum(-1) 
        
        stage_outputs.append({'elbo': importance_weights.detach(),
                              'log_prob_data': log_likelihood.detach(), 
                              'kl': kl.detach()
                              })

        return elbo_estimate

    ## END: Specific to Beta Variational Posterior AND CB likelihood ## 

class beta_beta_VAE(L.LightningModule):
    def __init__(self, 
                 hidden_dim=500, 
                 latent_dim=20,
                 var_link_func='softplus',
                 keep_prob=0.9,
                 dataset='mnist',
                 learning_rate=1e-3
                 ):
        super(beta_beta_VAE, self).__init__()

        assert dataset in ['mnist', 'cifar10'], f'unrecognized dataset: {dataset}'
        self.save_hyperparameters()

        self.n_output_encoder = 2 # beta
        self.n_output_decoder = 2 # beta
        if dataset == 'mnist':
            self.encoder = encoder_mnist(input_dim=int(28*28), hidden_dim=hidden_dim, latent_dim=latent_dim, keep_prob=keep_prob, n_output=self.n_output_encoder)
            self.decoder = decoder_mnist(input_dim=int(28*28), hidden_dim=hidden_dim, latent_dim=latent_dim, keep_prob=keep_prob, n_output=self.n_output_decoder)
        else:
            self.encoder = EncoderCIFAR(
                n_output=self.n_output_encoder, # for gaussian, KS, beta, etc
                latent_dim=latent_dim)
            self.decoder = DecoderCIFAR(n_output=self.n_output_decoder, latent_dim=latent_dim)
        # Initialize lists to store outputs for logging
        # https://lightning.ai/releases/2.0.0/#bc-changes-pytorch
        self.train_outputs = []
        self.val_outputs = []
        self.test_outputs = []

    ## START: Common to ALL VAEs ##   

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, 'train', self.train_outputs)
    
    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            fig = self.visualize_reconstructions(train_or_test='test', num_images=10)
            self.logger.experiment.log({"test reconstructions": wandb.Image(fig)})
            plt.close(fig)

            fig = self.visualize_reconstructions(train_or_test='train', num_images=10)
            self.logger.experiment.log({"train reconstructions": wandb.Image(fig)})
            plt.close(fig)
        return self.shared_step(batch, batch_idx, 'val', self.val_outputs)
    
    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, 'test', self.test_outputs)
    
    def shared_epoch_end(self, stage, stage_outputs):
        # Calculate average metrics
        avg_elbo = torch.stack([x['elbo'] for x in stage_outputs]).mean()
        avg_log_prob_data = torch.stack([x['log_prob_data'] for x in stage_outputs]).mean()
        avg_kl = torch.stack([x['kl'] for x in stage_outputs]).mean()
        
        # Log the average metrics
        self.log(f'avg_elbo_{stage}', avg_elbo)
        self.log(f'avg_log_prob_data_{stage}', avg_log_prob_data)
        self.log(f'avg_kl_{stage}', avg_kl)
        
        # Clear the outputs for the next epoch
        stage_outputs.clear()

    def on_train_epoch_end(self):
        self.shared_epoch_end('train', self.train_outputs)
    
    def on_validation_epoch_end(self):
        self.shared_epoch_end('val', self.val_outputs)

    def on_test_epoch_end(self):
        self.shared_epoch_end('test', self.test_outputs)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
    
    ## END: Common to ALL VAEs ##

    ## START: Common to all Beta Variational Posterior models ##

    def encode(self, x):
        x_encoded = self.encoder(x)
        log_alpha, log_beta = x_encoded.chunk(2, dim=-1)
        if self.hparams.var_link_func == 'softplus':
            alpha, beta = 1e-6 + torch.nn.functional.softplus(log_alpha), 1e-6 + torch.nn.functional.softplus(log_beta)
        elif self.hparams.var_link_func == 'exp':
            alpha, beta = torch.exp(log_alpha), torch.exp(log_beta)
        return alpha, beta
    
    def reparameterize(self, alpha, beta):
        q = torch.distributions.Beta(alpha, beta)
        z = q.rsample()
        return z
    
    def forward(self, x):
        alpha, beta = self.encode(x)
        z = self.reparameterize(alpha, beta)
        return self.decode(z), alpha, beta
    
    ## END: Common to all Beta Variational Posterior models ##

    ## START: Common to Beta likelihood ##

    def decode(self, z):
        likelihood_params = self.decoder(z)
        log_alpha, log_beta = likelihood_params.chunk(2, dim=-1)
        alpha, beta = 1e-6 + torch.nn.functional.softplus(log_alpha), 1e-6 + torch.nn.functional.softplus(log_beta)
        return (alpha, beta)

    def visualize_reconstructions(self, train_or_test, num_images=10):
        # sample images
        self.eval()
        loader = self.trainer.datamodule.train_dataloader() if train_or_test == 'train' else self.trainer.val_dataloaders
        x, _ = next(iter(loader)) # first batch
        x = x.view(x.size(0), -1)
        x = x.to(self.device)
        with torch.no_grad():
            likelihood_params, _, _ = self(x)
        alpha, beta = likelihood_params

        # reshape
        size = 28 if self.hparams.dataset == 'mnist' else 32
        channels = 1 if self.hparams.dataset == 'mnist' else 3
        alpha = alpha.view(-1, channels, size, size)
        beta = beta.view(-1, channels, size, size)
        x = x.view(-1, channels, size, size)

        x_hat_mean = torch.distributions.Beta(alpha, beta).mean

        x = x.cpu()
        x_hat_mean = x_hat_mean.cpu()
        alpha = alpha.cpu()
        beta = beta.cpu()

        num_rows = 4 if self.hparams.dataset == 'mnist' else 2
        fig, axes = plt.subplots(num_rows, num_images, figsize=(num_images, 4))
        for i in range(num_images):
            if self.hparams.dataset == 'mnist':
                axes[0, i].imshow(x[i].squeeze(), cmap='gray', vmin=0, vmax=1)
                axes[0, i].axis('off')
                axes[1, i].imshow(x_hat_mean[i].squeeze(), cmap='gray', vmin=0, vmax=1)
                axes[1, i].axis('off')
                axes[2, i].imshow(alpha[i].squeeze(), vmin=0, vmax=20)
                axes[2, i].set_title(f'{alpha[i].min().item():.2f} : {alpha[i].max().item():.2f}', fontsize=6, pad=-5)
                axes[2, i].axis('off')
                axes[3, i].imshow(beta[i].squeeze(), vmin=0, vmax=20)
                axes[3, i].axis('off')
                axes[3, i].set_title(f'{beta[i].min().item():.2f} : {beta[i].max().item():.2f}', fontsize=6, pad=-5)
                axes[3, i].axis('off')
                # add single combined colorbar for alpha/beta
                if i == 0:
                    fig.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=colors.Normalize(vmin=0, vmax=20)), ax=axes[2:4, :], orientation='horizontal')
            else:
                # cifar10
                img = np.transpose(x[i].numpy(), (1, 2, 0))
                img_hat = np.transpose(x_hat_mean[i].numpy(), (1, 2, 0))
                axes[0, i].imshow(img)
                axes[0, i].axis('off')
                axes[1, i].imshow(img_hat)
                axes[1, i].axis('off')
        return fig
    
    ## END: Common to Beta likelihood ##
    
    ## START: Specific to Beta Variational Posterior AND Beta likelihood ##

    def shared_step(self, batch, batch_idx, stage, stage_outputs, log_figure=True):
        x, _ = batch

        likelihood_params, alpha, beta = self(x)
        alpha_likelihood, beta_likelihood = likelihood_params
        alpha_likelihood = alpha_likelihood.view(x.size(0), -1)
        beta_likelihood = beta_likelihood.view(x.size(0), -1)

        x = x.view(x.size(0), -1)

        log_prob_data = torch.distributions.Beta(alpha_likelihood, beta_likelihood).log_prob(x).sum(-1)
        
        # kl between Beta(alpha, beta) and U(0, 1).
        kl = - torch.distributions.Beta(alpha, beta).entropy().sum(-1) 

        elbo = log_prob_data - kl
        neg_elbo = -elbo
        neg_elbo_mean = neg_elbo.mean()
        self.log(f'elbo_{stage}', -neg_elbo_mean, prog_bar=True)
        self.log(f'kl_{stage}', kl.mean())
        self.log(f'log_prob_data_{stage}', log_prob_data.mean())

        stage_outputs.append({'elbo': elbo.detach(), 'log_prob_data': log_prob_data.detach(), 'kl': kl.detach()})

        with torch.no_grad():
            log_alpha, log_beta = torch.log(1e-16 + alpha), torch.log(1e-16 + beta)
            logging_beta_latent_space(self, stage, batch_idx, log_alpha, log_beta, log_figure)

        return neg_elbo_mean
    
    def shared_step_iwae(self, batch, batch_idx, stage, stage_outputs, log_figure=True, k=200):
        x, _ = batch

        alpha, beta  = self.encode(x)
        z = torch.distributions.Beta(alpha, beta).rsample((k,))
        # decode
        alpha_likelihood, beta_likelihood = self.decode(z.view(-1, z.shape[-1])) #self.decode(z)
        alpha_likelihood, beta_likelihood = alpha_likelihood.view(k, x.size(0), -1), beta_likelihood.view(k, x.size(0), -1)
        

        # ELBO \approx \frac{1}{K} \sum_{k=1}^K ( log p(x | z_k) + log p(z_k) - log q(z_k | x) ), z_k \sim q(z|x)
        log_likelihood = torch.distributions.Beta(alpha_likelihood, beta_likelihood).log_prob(x).sum(-1)
        log_prior = torch.distributions.Uniform(low=0, high=1).log_prob(z).sum(-1) # uniform normal prior over U(0, 1)^d
        log_var_post = torch.distributions.Beta(alpha, beta).log_prob(z).sum(-1) # encoded var post

        # shape: (k, batch)
        log_importance_weights = log_likelihood + log_prior - log_var_post
        
        importance_weights = torch.logsumexp(log_importance_weights, 0)

        elbo_estimate = importance_weights.mean() # iwae elbo estimate for EACH image sample
        
        # kl between Beta(alpha, beta) and U(0, 1). Using torch implementation, no instability.
        kl = - torch.distributions.Beta(alpha, beta).entropy().sum(-1) 
        
        stage_outputs.append({'elbo': importance_weights.detach(),
                              'log_prob_data': log_likelihood.detach(), 
                              'kl': kl.detach()
                              })

        return elbo_estimate


    ## END: Specific to Beta Variational Posterior AND Beta likelihood ## 
    
class beta_ks_VAE(L.LightningModule):
    def __init__(self, 
                 hidden_dim=500, 
                 latent_dim=20, 
                 var_link_func='softplus',
                 keep_prob=0.9,
                 dataset='mnist',
                 learning_rate=1e-3,
                ):
        super(beta_ks_VAE, self).__init__()

        assert dataset in ['mnist', 'cifar10'], f'unrecognized dataset: {dataset}'
        self.save_hyperparameters()

        self.n_output_encoder = 2 # beta
        self.n_output_decoder = 2 # ks
        if dataset == 'mnist':
            self.encoder = encoder_mnist(input_dim=int(28*28), hidden_dim=hidden_dim, latent_dim=latent_dim, keep_prob=keep_prob, n_output=self.n_output_encoder)
            self.decoder = decoder_mnist(input_dim=int(28*28), hidden_dim=hidden_dim, latent_dim=latent_dim, keep_prob=keep_prob, n_output=self.n_output_decoder)
        else:
            self.encoder = EncoderCIFAR(
                n_output=self.n_output_encoder, # for gaussian, KS, beta, etc
                latent_dim=latent_dim)
            self.decoder = DecoderCIFAR(
                n_output=self.n_output_decoder, 
                latent_dim=latent_dim)
        # Initialize lists to store outputs for logging
        # https://lightning.ai/releases/2.0.0/#bc-changes-pytorch
        self.train_outputs = []
        self.val_outputs = []
        self.test_outputs = []
    
    ## START: Common to ALL VAEs ##   

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, 'train', self.train_outputs)
    
    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            fig = self.visualize_reconstructions(train_or_test='test', num_images=10)
            self.logger.experiment.log({"test reconstructions": wandb.Image(fig)})
            plt.close(fig)

            fig = self.visualize_reconstructions(train_or_test='train', num_images=10)
            self.logger.experiment.log({"train reconstructions": wandb.Image(fig)})
            plt.close(fig)
        return self.shared_step(batch, batch_idx, 'val', self.val_outputs)
    
    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, 'test', self.test_outputs)
    
    def shared_epoch_end(self, stage, stage_outputs):
        # Calculate average metrics
        avg_elbo = torch.stack([x['elbo'] for x in stage_outputs]).mean()
        avg_log_prob_data = torch.stack([x['log_prob_data'] for x in stage_outputs]).mean()
        avg_kl = torch.stack([x['kl'] for x in stage_outputs]).mean()
        
        # Log the average metrics
        self.log(f'avg_elbo_{stage}', avg_elbo)
        self.log(f'avg_log_prob_data_{stage}', avg_log_prob_data)
        self.log(f'avg_kl_{stage}', avg_kl)
        
        # Clear the outputs for the next epoch
        stage_outputs.clear()

    def on_train_epoch_end(self):
        self.shared_epoch_end('train', self.train_outputs)
    
    def on_validation_epoch_end(self):
        self.shared_epoch_end('val', self.val_outputs)

    def on_test_epoch_end(self):
        self.shared_epoch_end('test', self.test_outputs)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
    
    ## END: Common to ALL VAEs ##


    ## START: Common to all Beta Variational Posterior models ##

    def encode(self, x):
        x_encoded = self.encoder(x)
        log_alpha, log_beta = x_encoded.chunk(2, dim=-1)
        if self.hparams.var_link_func == 'softplus':
            alpha, beta = 1e-6 + torch.nn.functional.softplus(log_alpha), 1e-6 + torch.nn.functional.softplus(log_beta)
        elif self.hparams.var_link_func == 'exp':
            alpha, beta = torch.exp(log_alpha), torch.exp(log_beta)
        return alpha, beta
    
    def reparameterize(self, alpha, beta):
        q = torch.distributions.Beta(alpha, beta)
        z = q.rsample()
        return z
    
    def forward(self, x):
        alpha, beta = self.encode(x)
        z = self.reparameterize(alpha, beta)
        return self.decode(z), alpha, beta
    
    ## END: Common to all Beta Variational Posterior models ##

    
    ## START: Common to KS likelihood ##

    def decode(self, z):
        likelihood_params = self.decoder(z)
        log_a, log_b = likelihood_params.chunk(2, dim=-1)
        log_a, log_b = 5 * log_a, 10 * log_b # works without scaling, but scaling allows for more reasonable activations
        return (log_a, log_b)

    def visualize_reconstructions(self, train_or_test, num_images=10):
        # sample images
        self.eval()
        loader = self.trainer.datamodule.train_dataloader() if train_or_test == 'train' else self.trainer.val_dataloaders
        x, _ = next(iter(loader)) # first batch
        x = x.view(x.size(0), -1)
        x = x.to(self.device)
        with torch.no_grad():
            likelihood_params, _, _ = self(x)
        log_a, log_b = likelihood_params

        # reshape
        size = 28 if self.hparams.dataset == 'mnist' else 32
        channels = 1 if self.hparams.dataset == 'mnist' else 3
        log_a = log_a.view(-1, channels, size, size)
        log_b = log_b.view(-1, channels, size, size)
        x = x.view(-1, channels, size, size)

        # sample instead of closed form mean due to numerical instability
        x_hat_mean = KumaraswamyStable(log_a, log_b).sample((100,)).mean(0) # out of memory at 500 samples with bs 500

        x = x.to('cpu')
        x_hat_mean = x_hat_mean.to('cpu')
        log_a = log_a.cpu()
        log_b = log_b.cpu()

        lo, hi = -2.5, 7 # empirically find this is a reasonable range for log_a and log_b for visualization
        num_rows = 4 if self.hparams.dataset == 'mnist' else 2
        fig, axes = plt.subplots(num_rows, num_images, figsize=(num_images, 4))
        for i in range(num_images):
            if self.hparams.dataset == 'mnist':
                axes[0, i].imshow(x[i].squeeze(), cmap='gray', vmin=0, vmax=1)
                axes[0, i].axis('off')
                axes[1, i].imshow(x_hat_mean[i].squeeze(), cmap='gray', vmin=0, vmax=1)
                axes[1, i].axis('off')
                axes[2, i].imshow(log_a[i].squeeze(), vmin=lo, vmax=hi)
                axes[2, i].set_title(f'{log_a[i].min().item():.2f} : {log_a[i].max().item():.2f}', fontsize=6, pad=-5)
                axes[2, i].axis('off')
                axes[3, i].imshow(log_b[i].squeeze(), vmin=lo, vmax=hi)
                axes[3, i].set_title(f'{log_b[i].min().item():.2f} : {log_b[i].max().item():.2f}', fontsize=6, pad=-5)
                axes[3, i].axis('off')
                # add single combined colorbar for alpha/beta
                if i == 0:
                    fig.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=colors.Normalize(vmin=lo, vmax=hi)), ax=axes[2:4, :], orientation='horizontal')
            else:
                img = np.transpose(x[i].numpy(), (1, 2, 0))
                img_hat = np.transpose(x_hat_mean[i].numpy(), (1, 2, 0))
                axes[0, i].imshow(img)
                axes[0, i].axis('off')
                axes[1, i].imshow(img_hat)
                axes[1, i].axis('off')

        return fig

    ## END: Common to KS likelihood ##


    ## START: Specific to Beta Variational Posterior AND KS likelihood ##

    def shared_step(self, batch, batch_idx, stage, stage_outputs, log_figure=True):
        x, _ = batch

        likelihood_params, alpha, beta = self(x)
        log_a, log_b = likelihood_params
        log_a = log_a.view(x.size(0), -1)
        log_b = log_b.view(x.size(0), -1)

        x = x.view(x.size(0), -1)

        log_prob_data = KumaraswamyStable(log_a, log_b).log_prob(x, max_grad_log_a_clamp=0.2).sum(-1)
        
        # kl between Beta(alpha, beta) and U(0, 1).
        kl = - torch.distributions.Beta(alpha, beta).entropy().sum(-1) 

        elbo = log_prob_data - kl
        neg_elbo = -elbo
        neg_elbo_mean = neg_elbo.mean()
        self.log(f'elbo_{stage}', -neg_elbo_mean, prog_bar=True)
        self.log(f'kl_{stage}', kl.mean())
        self.log(f'log_prob_data_{stage}', log_prob_data.mean())

        stage_outputs.append({'elbo': elbo.detach(), 'log_prob_data': log_prob_data.detach(), 'kl': kl.detach()})

        ### logging for Beta latent space ##
        with torch.no_grad():
            log_alpha, log_beta = torch.log(1e-16 + alpha), torch.log(1e-16 + beta)
            logging_beta_latent_space(self, stage, batch_idx, log_alpha, log_beta, log_figure)

        return neg_elbo_mean
    
    def shared_step_iwae(self, batch, batch_idx, stage, stage_outputs, log_figure=True, k=200):
        x, _ = batch

        alpha, beta  = self.encode(x)
        z = torch.distributions.Beta(alpha, beta).rsample((k,))
        # decode
        log_a_likelihood, beta_likelihood = self.decode(z.view(-1, z.shape[-1])) #self.decode(z)
        log_a_likelihood, beta_likelihood = log_a_likelihood.view(k, x.size(0), -1), beta_likelihood.view(k, x.size(0), -1)

        # ELBO \approx \frac{1}{K} \sum_{k=1}^K ( log p(x | z_k) + log p(z_k) - log q(z_k | x) ), z_k \sim q(z|x)
        log_likelihood = KumaraswamyStable(log_a_likelihood, log_b_likelihood).log_prob(x).sum(-1)
        log_prior = torch.distributions.Uniform(low=0, high=1).log_prob(z).sum(-1) # uniform normal prior over U(0, 1)^d
        log_var_post = torch.distributions.Beta(alpha, beta).log_prob(z).sum(-1) # encoded var post

        # shape: (k, batch)
        log_importance_weights = log_likelihood + log_prior - log_var_post
        
        importance_weights = torch.logsumexp(log_importance_weights, 0)

        elbo_estimate = importance_weights.mean() # iwae elbo estimate for EACH image sample
        
        # kl between Beta(alpha, beta) and U(0, 1). Using torch implementation, no instability.
        kl = - torch.distributions.Beta(alpha, beta).entropy().sum(-1) 
        
        stage_outputs.append({'elbo': importance_weights.detach(),
                              'log_prob_data': log_likelihood.detach(), 
                              'kl': kl.detach()
                              })

        return elbo_estimate


    ## END: Specific to Beta Variational Posterior AND KS likelihood ## 