import os
import time
import warnings

import yaml
import time
import argparse
import os

import torch
import torch.nn.functional as F

from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import train_test_split_edges

from gnn import GCNEncoder, GAE, VGAE, VariationalMLPDecoder, MLPDecoder, posterior_predictive_metrics
warnings.filterwarnings("ignore", category=UserWarning, message="'train_test_split_edges' is deprecated")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch") # supress annoying warning which is internal to pytorch-geometric loading (I think)

from config import GNN_DIR, DATA_DIR
log_dir =  GNN_DIR + "logs/"
model_dir = GNN_DIR + "models/"

non_gpu_path_signature = 'your_name_abbreviation'


def encode_and_decode(model, x, train_pos_edge_index, neg_edge_index):
    embed = model(x, train_pos_edge_index)
    distrib_pos = model.decode(embed, train_pos_edge_index)
    distrib_neg = model.decode(embed, neg_edge_index)
    return distrib_pos, distrib_neg

def train(args):
    model.train()
    optimizer.zero_grad()
    
    neg_edge_index = negative_sampling(train_pos_edge_index, num_nodes=x.size(0))

    # decode and sample
    distrib_pos, distrib_neg = encode_and_decode(model, x, train_pos_edge_index, neg_edge_index)

    # binary cross entropy loss
    edge_prob_samples = torch.cat([distrib_pos.rsample(), distrib_neg.rsample()], dim=0).to(device)
    labels = torch.cat([torch.ones(train_pos_edge_index.shape[1]), torch.zeros(neg_edge_index.shape[1])], dim=0).to(device)
    recon_loss = F.binary_cross_entropy(edge_prob_samples, labels, reduction='mean').to(device)
    
    # ks and beta have closed form entropy, tanh requires sampling
    if args.latent_distrib in ['beta', 'ks']:
        distrib_pos_entropy = distrib_pos.entropy().sum()
        distrib_neg_entropy = distrib_neg.entropy().sum()
    elif args.latent_distrib == 'tanh-normal':
        distrib_pos_entropy = distrib_pos.entropy_estimate(num_samples=args.entropy_estimate_samples).sum() # takes mean over samples internally
        distrib_neg_entropy = distrib_neg.entropy_estimate(num_samples=args.entropy_estimate_samples).sum()
    else:
        raise ValueError(f"Invalid latent distribution: {args.latent_distrib}")

    # kl divergence
    kl_loss = - args.beta * (1 / (len(edge_prob_samples))) * (distrib_pos_entropy + distrib_neg_entropy)

    loss = recon_loss + kl_loss
    loss.backward()
    optimizer.step()

    metrics = {
        'recon_loss': float(recon_loss),
        'kl_loss': float(kl_loss),
        'loss': float(loss)
    }
    return metrics

def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        distrib_pos, distrib_neg = encode_and_decode(model, x, pos_edge_index, neg_edge_index)
        edge_prob_sample_pos = distrib_pos.rsample()
        edge_prob_sample_neg = distrib_neg.rsample()
    return model.test_edge(edge_prob_sample_pos, edge_prob_sample_neg) 

def test_posterior_predictive(model, pos_edge_index, neg_edge_index, num_samples=30, viz=False):
    model.eval()
    with torch.no_grad():
        post_pred_samples = []
        labels = torch.cat([torch.ones(pos_edge_index.shape[1]), torch.zeros(neg_edge_index.shape[1])], dim=0)

        for i in range(num_samples):
            # encode and decode
            distrib_pos, distrib_neg = encode_and_decode(model, x, pos_edge_index, neg_edge_index)

            # sample
            edge_prob_sample_pos = distrib_pos.rsample()
            edge_prob_sample_neg = distrib_neg.rsample()

            # sample the bernoulli likelihood
            post_pred_samples.append(torch.cat([
                torch.bernoulli(edge_prob_sample_pos), 
                torch.bernoulli(edge_prob_sample_neg)], 
                dim=0))
            
    post_pred_samples = torch.stack(post_pred_samples) # shape (num_samples, num_edges)
    metrics = posterior_predictive_metrics(post_pred_samples, labels)
    return metrics


if __name__ == "__main__":

    local_run = non_gpu_path_signature in os.getcwd()     
    config_path = GNN_DIR + 'config.yml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    parser = argparse.ArgumentParser(description='GNN Experiments')
    # data
    parser.add_argument('--dataset', type=str, choices=['Cora', 'Citeseer', 'Pubmed'], required=True, help='Dataset must be one of ["Cora", "Citeseer", "Pubmed"]')
    # model 
    parser.add_argument('--hidden_channels', type=int, default=config['hidden_channels'])
    parser.add_argument('--out_channels', type=int, default=config['out_channels'])
    parser.add_argument('--latent_distrib', type=str, choices=['beta', 'ks', 'tanh-normal'], required=True)
    parser.add_argument('--beta', type=float, required=True)
    parser.add_argument('--num_post_pred_samples', type=int, default=30)
    parser.add_argument('--entropy_estimate_samples', type=int, default=config['entropy_estimate_samples'])

    # training
    parser.add_argument('--learning_rate', type=float, default=config['learning_rate'])
    parser.add_argument('--epochs', type=int, default=config['epochs'])
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--run', type=int, required=True)
    args = parser.parse_args()


    # Set the device to GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    ## Load Data
    dataset = Planetoid(DATA_DIR, args.dataset, transform=T.NormalizeFeatures())
    #dataset.data
    data = dataset[0]
    data.train_mask = data.val_mask = data.test_mask = None
    torch.manual_seed(args.seed)
    data_tts = train_test_split_edges(data)
    
    model = GAE(
                GCNEncoder(dataset.num_features, args.hidden_channels, args.out_channels),
                VariationalMLPDecoder(args.out_channels, args.out_channels, latent_distrib=args.latent_distrib)
             )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    x = data.x.to(device)
    train_pos_edge_index = data.train_pos_edge_index.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print(f"Edge Latent: {args.latent_distrib} on {args.dataset} - Run {args.run}")


    times = []
    train_metrics = []
    test_metrics = []
    post_pred_test_metrics = []
    for epoch in range(0, args.epochs + 1):
        start_time = time.time()
        train_metrics_ = train(args)
        times.append(time.time() - start_time)
        train_metrics.append((epoch, train_metrics_))

        if epoch % 20 == 0:
            post_pred_test_metrics_ = test_posterior_predictive(model, data.test_pos_edge_index, data.test_neg_edge_index, viz=False, num_samples=args.num_post_pred_samples)
            post_pred_test_metrics.append((epoch, post_pred_test_metrics_))
        if epoch % 5 == 0:
            test_metrics_ = test(data.test_pos_edge_index, data.test_neg_edge_index)
            print('Epoch {:03d} ({:.1f} ms), AUC: {:.4f}, AP: {:.4f}, F1 {:.4f}'.format(epoch, times[-1] * 1e3, test_metrics_['auc'], test_metrics_['ap'], test_metrics_['f1']))
            test_metrics.append((epoch, test_metrics_))

    # save the logs
    filename = log_dir + f"edge_{args.latent_distrib}_{args.dataset}_{args.run}.pt"

    torch.save({
        'times': times,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'post_pred_test_metrics': post_pred_test_metrics
    }, filename)

    # save the model
    model_filename = model_dir + f"edge_{args.latent_distrib}_{args.dataset}_{args.run}.pt"
    torch.save(model.state_dict(), model_filename)