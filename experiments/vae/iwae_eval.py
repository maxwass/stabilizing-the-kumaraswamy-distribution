import torch
import lightning as L
from torch.utils.data import DataLoader
from vae import gauss_cb_VAE, gauss_beta_VAE, gauss_ks_VAE, ks_cb_VAE, ks_beta_VAE, ks_ks_VAE, beta_cb_VAE, beta_beta_VAE, beta_ks_VAE

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from kumaraswamy import KumaraswamyStable
from vae import MNISTDataModule, CIFAR10DataModule

import argparse
from datetime import datetime


from config import DATA_DIR, PROJECT_ROOT


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluate VAE')
    # model
    parser.add_argument('--var_post', type=str, required=True, default='gaussian', choices=['beta', 'gaussian', 'ks'], help='Variational posterior must be one of ["gaussian", "ks"]')
    parser.add_argument('--likelihood', type=str, required=True, default='cb', choices=['beta', 'cb', 'ks'], help='Likelihood must be one of ["beta", "cb", "ks"]')
    parser.add_argument('--var_link_func', type=str, default=None, choices=['softplus', 'exp'], help='Variational posterior link function (for the beta) must be one of ["softplus", "exp"]')
    # data
    parser.add_argument('--dataset', type=str, required=True, default='dataset', choices=['mnist', 'cifar10'], help='Dataset must be one of ["mnist", "cifar10"]')
    # metrics
    parser.add_argument('--k', type=int, default=15, help='Number of neighbors for KNN')
    # path to write results to 
    parser.add_argument('--path', type=str, default='./results_iwae/', help='Path to write results to')
    args = parser.parse_args()

    variational_posterior = args.var_post
    likelihood = args.likelihood
    dataset = args.dataset

    path_to_project = PROJECT_ROOT + '/experiments/vae/'
    path_to_models = path_to_project + 'trained_models/'

    model_path = path_to_models + f"{args.dataset}_{args.var_post}_{args.likelihood}.ckpt"
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

    model = model_class.load_from_checkpoint(model_path)

    print(model.hparams)

    if dataset == 'mnist':
        dm_class = MNISTDataModule
    elif dataset == 'cifar10':
        dm_class = CIFAR10DataModule

    dm = dm_class(batch_size=100, 
                  clamp_extreme_pixels=(likelihood in ['beta', 'ks']), 
                  data_dir=DATA_DIR, 
                  num_workers=0) # must be 0 in jupyter notebook
    
    # setup
    dm.prepare_data()
    dm.setup('fit')

    train_dataloader = dm.train_dataloader()
    val_dataloader = dm.val_dataloader()
    test_dataloader = dm.test_dataloader()

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
    
    ## Objective Evaluation: ELBO, KL, Reconstruction Loss

    print(f"Running Data through model...")
    train_outputs, test_outputs = [], []
    """
    for batch_idx, batch in enumerate(train_dataloader):
        if batch_idx % 10 == 0:
            print(batch_idx, '/', len(train_dataloader))
        if variational_posterior in ['ks', 'beta']:
            model.shared_step_iwae(batch, None, 'train', train_outputs, log_figure=False)
        else:
            model.shared_step_iwae(batch, None, 'train', train_outputs)
    """
    
    for batch_idx, batch in enumerate(test_dataloader):
        if batch_idx % 10 == 0:
            print(batch_idx, '/', len(test_dataloader))
        if variational_posterior in ['ks', 'beta']:
            model.shared_step_iwae(batch, batch_idx, 'test', test_outputs, log_figure=False)
        else:
            model.shared_step_iwae(batch, batch_idx, 'test', test_outputs)
    print(f"Done!")

    ## Compute ELBO, KL, and Reconstruction Loss
    def compute_objective_eval_metrics(outputs, split, print=False):
        elbo = [output['elbo'] for output in outputs]
        log_prob_data = [output['log_prob_data'] for output in outputs]
        kl = [output['kl'] for output in outputs]
        # elbo, kl, and log_prob_data are already summed over the latent dimension for each sample
        # Now stack the batches and then taking the mean/std
        elbo_mean, elbo_std = torch.stack(elbo).mean().item(), torch.stack(elbo).std().item()
        log_prob_data_mean, log_prob_data_std = torch.stack(log_prob_data).mean().item(), torch.stack(log_prob_data).std().item()
        kl_mean, kl_std = torch.stack(kl).mean().item(), torch.stack(kl).std().item()
        if print:
            print(f"** {split}: {likelihood} **")
            print(f"\tELBO: {elbo_mean:.3f} pm {elbo_std:3f}")
            print(f"\tlog likelihood: {log_prob_data_mean:.3f} pm {log_prob_data_std:3f}")
            print(f"\tKL: {kl_mean:.3f} pm {kl_std:3f}")
        return elbo_mean, elbo_std, log_prob_data_mean, log_prob_data_std, kl_mean, kl_std
    
    ## Compute KNN Accuracy
    def extract_latent_gaussian_representations(model, variational_posterior, dataloader):
        model.eval()
        latents = []
        labels = []
        with torch.no_grad():
            for x, y in dataloader:
                x = x.view(x.size(0), -1)
                if variational_posterior == 'gaussian':
                    mu, log_var = model.encode(x)
                    latents.append(mu)
                elif variational_posterior == 'ks':
                    log_a, log_b = model.encode(x)
                    latents.append(KumaraswamyStable(log_a, log_b).sample((1000,)).mean(dim=0))
                elif variational_posterior == 'beta':
                    alpha, beta = model.encode(x)
                    latents.append(torch.distributions.Beta(alpha, beta).mean)
                else:
                    raise ValueError(f"Variational posterior {variational_posterior} not supported")
                labels.append(y)
        latents = torch.cat(latents).cpu().numpy()
        labels = torch.cat(labels).cpu().numpy()
        return latents, labels

    def compute_knn_accuracy(model, variational_posterior, train_dataloader, test_dataloader, k=15):
        L.seed_everything(0)
        train_latents, train_labels = extract_latent_gaussian_representations(model, variational_posterior, train_dataloader)
        test_latents, test_labels = extract_latent_gaussian_representations(model, variational_posterior, test_dataloader)
        
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(train_latents, train_labels)
        test_preds = knn.predict(test_latents)
        
        accuracy = accuracy_score(test_labels, test_preds)
        return accuracy
    
    print(f"Dataset: {dataset}, Variational Posterior: {variational_posterior}, Likelihood: {likelihood}")
    #train_elbo_mean, train_elbo_std, train_log_prob_data_mean, trainlog_prob_data_std, train_kl_mean, train_kl_std = compute_objective_eval_metrics(train_outputs, 'Train')
    test_elbo_mean, test_elbo_std, test_log_prob_data_mean, test_log_prob_data_std, test_kl_mean, test_kl_std = compute_objective_eval_metrics(test_outputs, 'Test')
    knn_accuracy = compute_knn_accuracy(model, variational_posterior, train_dataloader, test_dataloader, k=args.k)
    print(f'KNN Test Accuracy: {knn_accuracy}')
    #print(f'Train ELBO: {train_elbo_mean:.3f} pm {train_elbo_std:.3f}')
    print(f"Test ELBO: {test_elbo_mean:.3f} pm {test_elbo_std:.3f}\n")

    # create a new file: args.path + f"{args.var_post}_{args.likelihood}_{args.dataset}.txt" and write the results to it
    results_file = f"{args.dataset}_{args.var_post}_{args.likelihood}.txt"
    with open(args.path + results_file, 'w') as file:
        file.write(f"Dataset: {dataset}, Variational Posterior: {variational_posterior}, Likelihood: {likelihood}\n")
        now = datetime.now()
        formatted_now = now.strftime("%A, %B %d, %Y at %I:%M %p")
        file.write(f"Date and Time of Eval: {formatted_now}\n")

        file.write(f"\nModel Info: \n{model.hparams}\n")

        #file.write("\n** Train **\n")
        #file.write(f"ELBO: {train_elbo_mean:.3f} pm {train_elbo_std:.3f}\n")
        #file.write(f"log likelihood: {train_log_prob_data_mean:.3f} pm {trainlog_prob_data_std:.3f}\n")
        #file.write(f"KL: {train_kl_mean:.3f} pm {train_kl_std:.3f}\n")
        file.write("\n** Test **\n")
        file.write(f"ELBO: {test_elbo_mean:.3f} pm {test_elbo_std:.3f}\n")
        file.write(f"log likelihood {test_log_prob_data_mean:.3f} pm {test_log_prob_data_std:.3f}\n")
        file.write(f"KL: {test_kl_mean:.3f} pm {test_kl_std:.3f}\n")
        file.write(f'KNN ({args.k}) Accuracy: {knn_accuracy:.5f}\n')

    print(f"Results written to: {args.path + results_file}")