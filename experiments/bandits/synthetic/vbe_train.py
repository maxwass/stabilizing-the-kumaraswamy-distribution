import yaml
import numpy as np
import time
import torch
import argparse
from neural_bandits import VariationalBanditEncoder
import os

from typing import List
import datetime

from config import BANDIT_DIR, DATA_DIR

non_gpu_path_signature = 'your_name_abbreviation' # 'maxw'
if non_gpu_path_signature == 'your_name_abbreviation':
    print(f"Warning: update gpu path signature")

def train(model, data, learning_rate, iterations, device, entropy_estimate_samples):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    X, probs = data
    X, probs = X.to(device), probs.to(device)


    times = []

    model.train()
    for i in range(iterations):
        step_start_time = time.time()
        optimizer.zero_grad()
        loss = model.training_step(X, probs, entropy_estimate_samples)
        loss.backward()
        optimizer.step()
        times.append(time.time() - step_start_time)

        metrics = model.metrics[-1]
        print(f"{i}: ", {k: round(v, 3) if isinstance(v, float) else v for k, v in metrics.items()}, f" time: {np.mean(times[-10:])*1e3:.1f} ms")
    
    model.eval()
    return times

def save_model(args, model, model_name, times):
    now = datetime.datetime.now()
    filename = model_name # + '_' + now.strftime('%d-%b-%Y_Hour-%H-Min-%M')
    model_dir = BANDIT_DIR + "synthetic/models" if non_gpu_path_signature in os.getcwd() else "./models"
    os.makedirs(model_dir, exist_ok=True)
    final_model_path = os.path.join(model_dir, f"{filename}.ckpt")
    
    # Save model state_dict and metrics
    torch.save({
        'args': args,
        'model_state_dict': model.state_dict(),
        'metrics': model.metrics,
        'step_times': times,
        'cumulative_reward': model.cumulative_reward,
        'cumulative_regret': model.cumulative_regret
    }, final_model_path)
    
    print(f'Model and metrics saved to {final_model_path}')



if __name__ == "__main__":

    local_run = non_gpu_path_signature in os.getcwd()     
    config_path = BANDIT_DIR + 'synthetic/vbe_config.yml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    parser = argparse.ArgumentParser(description='Neural Contextual Bernoulli Bandits')
    # data
    parser.add_argument('--data_dim', type=int, required=True, default=5, help='Dimension of the data')
    parser.add_argument('--data_power', type=int, required=True, default=5, help='Power to raise the normalized probabilities to. Makes the data more skewed toward 0, forces more exploration.')
    parser.add_argument('--num_bandits', type=int, required=True, default=config['num_bandits'])
    # model 
    parser.add_argument('--var_post', type=str, required=True, default='ks', choices=['beta', 'ks', 'tanh-normal', 'concrete'], help='Variational posterior must be one of ["beta", "ks", "tanh-normal", "concrete"]')
    parser.add_argument('--top_m', type=int, required=True, default=config['top_m'])
    parser.add_argument('--hidden_layers', type=List[int], default=config['hidden_layers'])
    parser.add_argument('--learning_rate', type=float, default=config['learning_rate'])
    #parser.add_argument('--entropy_scale', type=float, default=config['entropy_scale'])
    parser.add_argument('--entropy_estimate_samples', type=int, default=config['entropy_estimate_samples'])
    parser.add_argument('--iterations', required=True, type=int, default=config['iterations'])
    parser.add_argument('--seed', type=int, default=config['seed'])
    parser.add_argument('--run', type=int, default=0)
    args = parser.parse_args()


    # Set the device to GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    ## Load Data
    data_name = f"bandit_D_{args.data_dim}_K_{args.num_bandits}_P={args.data_power}"
    data_dir = DATA_DIR if local_run else './'
    X, probs = torch.load(f'{data_dir}/{data_name}.pt', weights_only=True)

    # Initialize the model
    model_name = f"numBandits_{args.num_bandits}_topM_{args.top_m}_varPost_{args.var_post}_run_{args.run}"
    torch.manual_seed(args.seed)
    model = VariationalBanditEncoder(
        input_dim=X.shape[1],
        hidden_layers=args.hidden_layers,
        num_bandits=X.shape[0],
        top_m=args.top_m,
        var_post=args.var_post,
    ).to(device)
    times = train(model, (X, probs), args.learning_rate, args.iterations, device, entropy_estimate_samples=args.entropy_estimate_samples)

    # Save the final model checkpoint
    args = dict(args.__dict__)
    print(args)
    #save_model(args, model, model_name, times) # UNCOMMENT THIS TO OVERWRITE THE MODEL FILE