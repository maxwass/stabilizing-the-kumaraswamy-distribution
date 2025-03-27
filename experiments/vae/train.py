import yaml
import lightning as L
import torch
import argparse
from argparse import ArgumentParser
from vae import gauss_cb_VAE, gauss_beta_VAE, gauss_ks_VAE, ks_cb_VAE, ks_beta_VAE, ks_ks_VAE, beta_cb_VAE, beta_beta_VAE, beta_ks_VAE
from vae import MNISTDataModule, CIFAR10DataModule
import datetime
import os

from lightning.pytorch.loggers import WandbLogger
import warnings

# Suppress the specific warning
warnings.filterwarnings("ignore", message="The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=7` in the `DataLoader` to improve performance.")
warnings.filterwarnings("ignore", message="The 'val_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=7` in the `DataLoader` to improve performance.")
warnings.filterwarnings("ignore", message="The 'test_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=7` in the `DataLoader` to improve performance.")

from config import DATA_DIR, PROJECT_ROOT
path_to_project = PROJECT_ROOT + '/experiments/vae/'
config_path = path_to_project + 'config.yml'
path_to_models = path_to_project + 'trained_models/'
non_gpu_path_signature = 'your_name_abbreviation'

if __name__ == "__main__":

    local_run = non_gpu_path_signature in os.getcwd()     
    with open(config_path, 'r') as file: # CHANGE BACK TO config.yml WHEN DONE TESTING
        config = yaml.safe_load(file)

    parser = argparse.ArgumentParser(description='Train VAE on MNIST or CIFAR10')
    parser.add_argument('--latent_dimension', type=int, required=True, help='Latent dimension must be 20 for MNIST and 50 for CIFAR10')
    parser.add_argument('--var_post', type=str, required=True, default='gaussian', choices=['beta', 'gaussian', 'ks'], help='Variational posterior must be one of ["beta", "gaussian", "ks"]')
    parser.add_argument('--likelihood', type=str, required=True, default='cb', choices=['beta', 'cb', 'ks'], help='Likelihood must be one of ["beta", "cb", "ks"]')
    parser.add_argument('--hidden_units', type=int, default=config['hidden_units'])
    parser.add_argument('--keep_prob', type=float, default=config['keep_prob'])
    parser.add_argument('--var_link_func', type=str, default=None, choices=['softplus', 'exp'], help='Variational posterior link function (for the beta) must be one of ["softplus", "exp"]')
    # training
    parser.add_argument('--learning_rate', type=float, default=config['learning_rate'])
    parser.add_argument('--batch_size', type=int, default=config['batch_size'])
    parser.add_argument('--max_epochs', type=int, default=config['max_epochs'])
    parser.add_argument('--seed', type=int, default=config['seed'])
    # data
    parser.add_argument('--dataset', type=str, required=True, default='dataset', choices=['mnist', 'cifar10'], help='Dataset must be one of ["mnist", "cifar10"]')
    args = parser.parse_args()

    if args.dataset == 'mnist':
        assert args.latent_dimension == 20, 'For MNIST, latent dimension must be 20'
    elif args.dataset == 'cifar10':
        assert args.latent_dimension == 50, 'For CIFAR10, latent dimension must be 50'


    ## Create DataModule
    clamp_extreme_pixels = (args.likelihood in ['beta', 'ks'])
    if args.dataset == 'mnist':
        dm_class = MNISTDataModule
    elif args.dataset == 'cifar10':
        dm_class = CIFAR10DataModule
    
    dm = dm_class(batch_size=args.batch_size, 
                  clamp_extreme_pixels=clamp_extreme_pixels, 
                  data_dir=DATA_DIR, 
                  num_workers=3 if torch.cuda.is_available() else 0)
    
    # setup
    dm.prepare_data()
    dm.setup('fit')


    ## Create Model
    if args.var_post == 'gaussian':
        if args.likelihood == 'cb':
            model_class = gauss_cb_VAE
        elif args.likelihood  == 'beta':
            model_class = gauss_beta_VAE
        elif args.likelihood  == 'ks':
            model_class = gauss_ks_VAE
    
    elif args.var_post == 'ks':
        if args.likelihood == 'cb':
            model_class = ks_cb_VAE
        elif args.likelihood  == 'beta':
            model_class = ks_beta_VAE
        elif args.likelihood  == 'ks':
            model_class = ks_ks_VAE
    
    elif args.var_post == 'beta':
        if args.likelihood == 'cb':
            model_class = beta_cb_VAE
        elif args.likelihood  == 'beta':
            model_class = beta_beta_VAE
        elif args.likelihood  == 'ks':
            model_class = beta_ks_VAE
    

    model_name = args.dataset + '_' + args.var_post + '_' + args.likelihood
    if local_run:
        # if running locally, dont log to wandb, i.e. dont use log_model="all" and do use offline=True
        wandb_logger = WandbLogger(name=model_name, project="Local-KumaraswamyVAE", offline=True)
    else:
        wandb_logger = WandbLogger(log_model="all", name=model_name, project="KumaraswamyVAE")

    L.seed_everything(args.seed)

    model_args = {'hidden_dim': args.hidden_units, 'latent_dim': args.latent_dimension, 'keep_prob': args.keep_prob, 'learning_rate': args.learning_rate, 'dataset': args.dataset}
    if args.var_post == 'beta':
        model_args['var_link_func'] = args.var_link_func
    
    print(f"\n\n**Training {model_name} with the following model args**")
    print('\t', model_args, '\n\n')
    
    model = model_class(**model_args)

    """
    model = model_class(
        hidden_dim=args.hidden_units,
        latent_dim=args.latent_dimension,
        keep_prob=args.keep_prob,
        learning_rate=args.learning_rate,
        dataset=args.dataset
        )
    """

    ## Create Trainer
    trainer = L.Trainer(
        logger=wandb_logger,
        max_epochs=args.max_epochs,
        accelerator='cuda' if torch.cuda.is_available() else 'cpu', #'auto', # this will be in current directory
        check_val_every_n_epoch=10,
        fast_dev_run=True, #False # local_run
    )

    trainer.fit(model, dm)

    # Save the final model checkpoint
    now = datetime.datetime.now()
    filename = model_name + '_' + now.strftime('%d-%b-Hour-%H-Min-%M')
    final_model_path = f"./models/{filename}.ckpt"
    trainer.save_checkpoint(final_model_path)
    print(f'Model saved to {final_model_path}')

    trainer.test(model, datamodule=dm)