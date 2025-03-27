import time
import math
from typing import List

from torch.optim import Optimizer
from torch import Tensor

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Beta, Bernoulli, RelaxedBernoulli

from kumaraswamy import KumaraswamyStable
from simple_squashed_normal import TanhNormal


def stable_bernoulli_log_prob(log_p, binary_rewards):
    log_q = torch.log1p(-torch.exp(log_p))  # log(1 - p) using log1p for stability
    log_prob = binary_rewards * log_p + (1 - binary_rewards) * log_q
    return log_prob

class VariationalBanditEncoder(nn.Module):
    def __init__(self, 
        input_dim: int, 
        hidden_layers: List[int], 
        num_bandits: int, 
        top_m: int, 
        var_post: str = 'ks'
        ):
        super().__init__()
        assert var_post in ['ks', 'beta', 'tanh-normal', 'concrete'], "Invalid variational posterior. Must be one of ['ks', 'beta', 'tanh-normal', 'concrete']"
        self.var_post = var_post
        #self.q = KumaraswamyStable if var_post == 'ks' else Beta
        if var_post == 'ks':
            self.q = KumaraswamyStable
        elif var_post == 'beta':
            self.q = Beta
        elif var_post == 'tanh-normal':
            self.q = TanhNormal
        elif var_post == 'concrete':
            self.q = RelaxedBernoulli
        
        # Define the shared MLP
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.LeakyReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 2))  # Output layer to get alpha and beta
        self.mlp = nn.Sequential(*layers)
        
        self.num_bandits = num_bandits
        self.top_m = top_m

        # keep track of the past context/reward pairs
        self.contexts = []
        self.rewards = []
        self.arm_indices = []
        # counter of size num_bandits to keep track of the number of times each arm is selected
        self.arm_counter = torch.zeros(num_bandits)
        self.arm_set = set()
        
        self.replay_buffer_arms = []
        self.replay_buffer_rewards = []
        # register buffer
        self.replay_buffer_arm_last_occurance = torch.ones(num_bandits, dtype=torch.int) * -1
        # ensure this is moved to GPU when model is moved to GPU
        #self.register_buffer('replay_buffer_arm_last_occurance', torch.ones(num_bandits) * -1)
        
        # Metrics
        self.cumulative_reward = 0.0
        self.cumulative_regret = 0.0
        self.step_count = 0
        self.start_time = time.time()

        self.metrics = []

        self.optimal_probs = None

    def forward_beta(self, x):
        output = self.mlp(x)  # Shape: [K, 2]
        pre_alpha, pre_beta = output.chunk(2, dim=1)
        alpha = F.softplus(pre_alpha) + 1e-6
        beta = F.softplus(pre_beta) + 1e-6
        #alpha = torch.exp(pre_alpha) + 1e-6 # cause unstable training - nans
        #beta = torch.exp(pre_beta) + 1e-6
        return alpha.squeeze(), beta.squeeze()

    def forward_concrete(self, x):
        output = self.mlp(x)
        pre_temp, logit = output.chunk(2, dim=1)
        temp = torch.exp(pre_temp) + 1e-6
        return temp.squeeze(), logit.squeeze()
    
    def forward_ks(self, x):
        output = self.mlp(x)
        log_a, log_b = output.chunk(2, dim=1)
        return log_a.squeeze(), log_b.squeeze()
    
    def forward_tanh_normal(self, x):
        output = self.mlp(x)
        mu, log_stdv = output.chunk(2, dim=1)
        return mu.squeeze(), log_stdv.squeeze()

    def forward(self, X):
        if self.var_post == 'beta':
            return self.forward_beta(X)
        elif self.var_post == 'ks':
            return self.forward_ks(X)
        elif self.var_post == 'tanh-normal':
            return self.forward_tanh_normal(X)
        elif self.var_post == 'concrete':
            return self.forward_concrete(X)
        else:
            raise ValueError("Invalid variational posterior. Must be one of ['beta', 'ks', 'tanh-normal']")

    def training_step(self, X, true_probs, entropy_estimate_samples):
        # X: [K, D], true_probs: [K]
        
        ### perform arm selection here ###
        # param_1/2 is log_a/log_b for Kumaraswamy, alpha/beta for Beta, and mu/log_stdv for TanhNormal
        var_post_param_1, var_post_param_2 = self.forward(X) # Each of shape: [num_bandits]
        if self.var_post in ['beta', 'ks']:
            distributions = self.q(var_post_param_1, var_post_param_2) # TODO: eventually implement low/high for ks
        if self.var_post == 'concrete':
            distributions = self.q(temperature=var_post_param_1, logits=var_post_param_2)
        elif self.var_post == 'tanh-normal':
            distributions = self.q(var_post_param_1, var_post_param_2, low=0, high=1)

        samples = distributions.rsample()  # Shape: [num_bandits]

        # Select top-M bandits based on sampled values
        _, top_m_sample_indices = torch.topk(samples, self.top_m)
        self.arm_set.update(top_m_sample_indices.tolist())
        
        # Get true probabilities for selected top-M bandits
        rewards = Bernoulli(true_probs[top_m_sample_indices]).sample()  # Shape: [M] # TODO: should these be python ints instead of tensors for efficiency?

        # save context/reward pairs
        self.replay_buffer_arms.extend(top_m_sample_indices.tolist())
        self.replay_buffer_rewards.extend(rewards)

        # update last occurance of each arm, which is the final M indices of the replay buffer: [len(replay_buffer_arms) - M, len(replay_buffer_arms) - M + 1, ...,  len(replay_buffer_arms) - 1]
        l = len(self.replay_buffer_arms)
        self.replay_buffer_arm_last_occurance[top_m_sample_indices] = torch.arange(l - self.top_m, l, dtype=torch.int)
                #list(range(len(self.replay_buffer_arms) - self.top_m, len(self.replay_buffer_arms)))
        # Update metrics
        self.optimal_probs = self.optimal_probs if self.optimal_probs is not None else true_probs.topk(self.top_m).values # expected reward if we always select the top m bandits
        selected_probs = true_probs[top_m_sample_indices]
        regret = (self.optimal_probs - selected_probs).mean()
    
        self.cumulative_reward += rewards.sum().item()
        self.cumulative_regret += regret.item()
        self.step_count += 1
    
        ### Compute ELBO ###
        
        ## Log prob
        # 1. Place sample from arm i to each location in the replay buffer where arm i was pulled.
        # ex) replay_buffer_arms = [4,     7,   7,   0,   9, ...]
        #     samples            = [s_0, s_1, s_2, s_3, s_4, ...]
        #     samples_rb         = [s_4, s_7, s_7, s_0, s_9, ...]
        samples_rb = samples[self.replay_buffer_arms] # samples are for each arm ... distribute to 

        # 2. For each reward in replay buffer, use the sample for that arm to parameterize the Bernoulli likelihood, and compute the log prob
        # of that reward under that sample.
        rb_rewards = torch.tensor(self.replay_buffer_rewards, dtype=torch.float32).to(X.device)
        #log_prob = stable_bernoulli_log_prob(samples_rb, rb_rewards).sum() # use with log_samples
        if self.var_post == 'concrete':
            samples_rb = samples_rb.clamp(1e-3, 1 - 1e-3)
        log_prob = Bernoulli(probs=samples_rb).log_prob(rb_rewards).sum()

        ## Entropy
        # For each arm that has been pulled, compute it's Shannon Entropy. 
        # NOTE: This is NOT the entropy of the entire replay buffer, but the entropy of the unique arms pulled.
        pulled_arms = torch.tensor(list(self.arm_set), dtype=torch.int).to(X.device)
        if self.var_post in ['beta', 'ks']:
            entropy = self.q(var_post_param_1[pulled_arms], var_post_param_2[pulled_arms]).entropy()
        elif self.var_post == 'tanh-normal':
            entropy = self.q(var_post_param_1[pulled_arms], var_post_param_2[pulled_arms]).entropy_estimate(num_samples=entropy_estimate_samples)
        elif self.var_post == 'concrete':
            # use num_samples to estimate temperature: entropy = -E_q[log q(x)] = - 1/L sum_{l=1}^L log q(x_l)
            c_samples = self.q(temperature=var_post_param_1[pulled_arms], logits=var_post_param_2[pulled_arms]).rsample(sample_shape=torch.Size([entropy_estimate_samples]))
            c_samples = c_samples.clamp(1e-3, 1 - 1e-3)
            #entropy = - self.q(temperature=var_post_param_1[pulled_arms], logits=var_post_param_2[pulled_arms]).log_prob(c_samples).mean(dim=0)
            # be robust to nan and inf:
            finite_mask = self.q(temperature=var_post_param_1[pulled_arms], logits=var_post_param_2[pulled_arms]).log_prob(c_samples).isfinite()
            entropy = - self.q(temperature=var_post_param_1[pulled_arms], logits=var_post_param_2[pulled_arms]).log_prob(c_samples).mean(dim=0) # CHECK THAT MEAN RED OVER CORRECT DIM
            if torch.isnan(entropy).any():
                print("Entropy is Nan!")
                print(f"var_post_param_1: {var_post_param_1[pulled_arms]}, var_post_param_2: {var_post_param_2[pulled_arms]}")
                print(f"Entropy: {entropy}")
                print(f"Samples: {c_samples = }")
        assert len(entropy) == len(self.arm_set), f"# entropy terms != number of unique arms pulled: len(entropy): {len(entropy)}, len(self.arm_set): {len(self.arm_set)}"

        # decreasing c increases the entropy penalty, promoting exploration. c=1.0 is the default, and corresponds to a mean of the entropies.
        c = 1.0 
        entropy_scale = (1/len(entropy))**(c) 
        entropy = entropy_scale * entropy.sum()
        elbo = log_prob + entropy#.mean() # implicitly using a entropy_scaling of 1/len(entropy) == 1/num_unique_arms_pulled
        loss = -elbo


        # Log metrics
        metrics = {
            'reward': rewards.sum().item(), 
            'regret': regret.item(),
            'cumulative_reward': self.cumulative_reward,
            'cumulative_regret': self.cumulative_regret,
            'arms': len(self.arm_set),
            'elbo': elbo.mean().item(),
            'log_prob': log_prob.item(),
            'entropy': entropy.sum().item(),
            }
        self.metrics.append(metrics)
        return loss


###### Langevin Monte Carlo ######

# From author implementation in "Langevin Monte Carlo for Thompson Sampling", https://github.com/devzhk/LMCTS/blob/97114fc7a2160ba5d45c9ef483d2284497f81be6/algo/langevin.py
# example how to train: https://github.com/devzhk/LMCTS/blob/97114fc7a2160ba5d45c9ef483d2284497f81be6/train_utils/helper.py#L59
def lmc(params: List[Tensor],
        d_p_list: List[Tensor],
        weight_decay: float,
        lr: float):
    r"""Functional API that performs Langevine MC algorithm computation.
    """

    for i, param in enumerate(params):
        d_p = d_p_list[i]
        if weight_decay != 0:
            d_p = d_p.add_(param, alpha=weight_decay)

        param.add_(d_p, alpha=-lr)

class LangevinMC(Optimizer):
    def __init__(self,
                 params,              # parameters of the model
                 lr=0.01,             # learning rate
                 beta_inv=0.01,       # inverse temperature parameter
                 sigma=1.0,           # variance of the Gaussian noise
                 weight_decay=1.0,
                 device=None):   # l2 penalty
        if lr < 0:
            raise ValueError('lr must be positive')
        if device:
            self.device = device
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.beta_inv = beta_inv
        self.lr = lr
        self.sigma = sigma
        self.temp = - math.sqrt(2 * beta_inv / lr) * sigma
        self.curr_step = 0
        defaults = dict(weight_decay=weight_decay)
        super(LangevinMC, self).__init__(params, defaults)

    def init_map(self):
        self.mapping = dict()
        index = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    num_param = p.numel()
                    self.mapping[p] = [index, num_param]
                    index += num_param
        self.total_size = index

    @torch.no_grad()
    def step(self):
        self.curr_step += 1
        if self.curr_step == 1:
            self.init_map()

        lr = self.lr
        temp = self.temp
        noise = temp * torch.randn(self.total_size, device=self.device)

        for group in self.param_groups:
            weight_decay = group['weight_decay']

            params_with_grad = []
            d_p_list = []
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)

                    start, length = self.mapping[p]
                    add_noise = noise[start: start + length].reshape(p.shape)
                    delta_p = p.grad
                    delta_p = delta_p.add_(add_noise)
                    d_p_list.append(delta_p)
                    # p.add_(delta_p)
            lmc(params_with_grad, d_p_list, weight_decay, lr)
class LMCTS(nn.Module):
    def __init__(self, 
        input_dim: int, 
        hidden_layers: List[int], 
        num_bandits: int, 
        top_m: int, 
        ):
        super().__init__()
        
        # Define the shared MLP
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.LeakyReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))  # Output layer to expected reward
        self.mlp = nn.Sequential(*layers)
        
        self.num_bandits = num_bandits
        self.top_m = top_m

        # keep track of the past context/reward pairs
        self.contexts = []
        self.rewards = []
        self.arm_set = set()
        
        # Metrics
        self.cumulative_reward = 0.0
        self.cumulative_regret = 0.0
        self.step_count = 0
        self.start_time = time.time()

        self.metrics = []

    def forward(self, X):
        return self.mlp(X) # log expected reward

    def training_step(self, X, true_probs, optimizer, inner_num_iters):
        # X: [K, D], true_probs: [K]
        
        logits = self.forward(X).squeeze()  # Each of shape: [K]
        
        # Select top-M bandits based on logits
        _, top_m_sample_indices = torch.topk(logits, self.top_m)
        
        # Get true probabilities for selected top-M bandits
        rewards = Bernoulli(true_probs[top_m_sample_indices]).sample()  # Shape: [M]

        X_top_m = X[top_m_sample_indices]

        # save context/reward pairs
        self.contexts.append(X_top_m)
        self.rewards.append(rewards)
        self.arm_set.update(top_m_sample_indices.tolist())

        # concatenate all contexts and rewards for training
        X_train = torch.cat(self.contexts, dim=0).to(X.device)
        rewards_train = torch.cat(self.rewards, dim=0).to(X.device)

        # turn on training mode
        # save old weights
        last_weights = [param.clone() for param in self.mlp.parameters()]
        self.mlp.train()
        nan_count = 0
        for i in range(inner_num_iters):
            # langevin monte carlo
            self.mlp.zero_grad()
            logits = self.forward(X_train)
            
            # ensure logits and rewards have the same shape. top_m == 1 causes shape issue.
            if logits.ndim > rewards_train.ndim:
                rewards_train = rewards_train.unsqueeze(-1)

            # torch binary cross entropy. See line 54 for reduction: https://github.com/devzhk/LMCTS/blob/97114fc7a2160ba5d45c9ef483d2284497f81be6/train_utils/helper.py#L265
            loss = F.binary_cross_entropy_with_logits(logits, rewards_train, reduction='sum') # tried mean, and it didnt work!
            if torch.isnan(loss):
                nan_count += 1
                print("Loss is Nan!...replacing weights with pre-LMC weights")
                # replace the weights with the last stable weights
                with torch.no_grad():
                    for param, old_param in zip(self.mlp.parameters(), last_weights):
                        param.data = old_param.data
                    if nan_count > 5:
                        # add some tiny noise to the weights
                        print("5 Monte Carlo steps with Nan loss. Adding N(0, 1) * 1e-4 noise to pre-MC weights.")
                        for param in self.mlp.parameters():
                            param.data += torch.randn_like(param) * 1e-4
                #break
            else:
                loss.backward()
                #last_weights = [param.clone() for param in self.mlp.parameters()]
            optimizer.step() # LMC is implemented in this custom optimizer
        #assert not torch.isnan(loss), "Loss is Nan!"

        self.mlp.eval()
        
        # Update metrics
        optimal_probs = true_probs.topk(self.top_m).values # reward if we always select the top m bandits
        regret = (optimal_probs - true_probs[top_m_sample_indices]).sum()
        
        self.cumulative_reward += rewards.sum().item()
        self.cumulative_regret += regret.sum().item()
        self.step_count += 1
        
        # Log metrics
        metrics = {
            'reward': rewards.sum().item(), 
            'regret': regret.item(),
            'log_prob': - loss.mean().item(),
            'arms': len(self.arm_set),
            'nan_count': nan_count,
            'cumulative_reward': self.cumulative_reward,
            'cumulative_regret': self.cumulative_regret
        }
        self.metrics.append(metrics)
        return