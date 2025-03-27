import yaml
import numpy as np
import math
import time
import torch
import argparse
from neural_bandits import LMCTS, LangevinMC
import os

from config import BANDIT_DIR, DATA_DIR

from typing import List

non_gpu_path_signature = 'your_name_abbreviation'

def train(model, data, learning_rate, beta_inv, iterations, inner_num_iters, device):
    # see line 47: https://github.com/devzhk/LMCTS/blob/97114fc7a2160ba5d45c9ef483d2284497f81be6/train_utils/helper.py#L59

    dim_context = data[0].shape[1]
    beta_inv = beta_inv * dim_context * math.log(iterations)
    optimizer = LangevinMC(model.parameters(), 
                           lr=learning_rate, 
                           beta_inv=beta_inv, 
                           weight_decay=2.0,
                           device=device)
    X, probs = data
    X, probs = X.to(device), probs.to(device)


    times = []

    for i in range(iterations):
        step_start_time = time.time()
        _ = model.training_step(X, probs, optimizer, inner_num_iters) # optimization happening inside
        times.append(time.time() - step_start_time)
        metrics = model.metrics[-1]
        print(f"{i}: ", {k: round(v, 3) if isinstance(v, float) else v for k, v in metrics.items()}, f" time: {np.mean(times[-10:])*1e3:.1f} ms")
    
    return times

def save_model(args, model, model_name, times):
    filename = model_name
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
    config_path = BANDIT_DIR + 'synthetic/lmcts_config.yml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    parser = argparse.ArgumentParser(description='Lagevin Monte Carlo Neural Contextual Bernoulli Bandits')
    # data
    parser.add_argument('--data_dim', type=int, required=True, default=5, help='Dimension of the data')
    parser.add_argument('--data_power', type=int, required=True, default=5, help='Power to raise the normalized probabilities to. Makes the data more skewed toward 0, forces more exploration.')
    parser.add_argument('--num_bandits', type=int, required=True, default=config['num_bandits'])
    # model 
    parser.add_argument('--top_m', type=int, required=True, default=config['top_m'])
    parser.add_argument('--hidden_layers', type=List[int], default=config['hidden_layers'])
    # optimization
    parser.add_argument('--learning_rate', type=float, default=config['learning_rate'])
    parser.add_argument('--iterations', required=True, type=int, default=config['iterations']) # outer loop
    # langevin monte carlo specific
    parser.add_argument('--inner_num_iters', required=True, type=int, default=config['inner_num_iters']) # K in the paper
    parser.add_argument('--beta_inv', type=float, default=config['beta_inv'])
    # reproducibility
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
    model_name = f"LMCTS_numBandits_{args.num_bandits}_topM_{args.top_m}_run_{args.run}"
    torch.manual_seed(args.seed)

    model = LMCTS(
        input_dim=X.shape[1],
        hidden_layers=args.hidden_layers,
        num_bandits=X.shape[0],
        top_m=args.top_m,
    ).to(device)
    times = train(model, (X, probs), args.learning_rate, args.beta_inv, args.iterations, args.inner_num_iters, device)

    # Save the final model checkpoint
    args = dict(args.__dict__)
    print(args)
    save_model(args, model, model_name, times)