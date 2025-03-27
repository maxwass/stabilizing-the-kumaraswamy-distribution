import math
import torch
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all
from torch import exp, log
from typing import Optional
from kumaraswamy import log1mexp
from typing import Union
from numbers import Number
import numpy as np
#from pyro.distributions.torch_distribution import TorchDistributionMixin


class RelaxedBernoulli(Distribution):
    """
    Args:
        temperature (Tensor): Positive temperature parameter.
        logits (Tensor): Logits parameter (log-odds of success).
    """
    arg_constraints = {
        "temperature": constraints.positive,
        "logits": constraints.real,
    }

    has_rsample = True
    
    def __init__(self, 
                 temperature: torch.Tensor, 
                 logits: torch.Tensor, 
                 validate_args=None):
        self.temperature, self.logits = broadcast_all(temperature, logits)
        
        # We do NOT subclass TransformedDistribution. We manually implement the rsample method.

        device = self.temperature.device
        self.batch_shape_ = self.temperature.size() 
        super().__init__(batch_shape=self.temperature.size(), validate_args=validate_args) #validate_args=validate_args)
    
    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(RelaxedBernoulli, _instance)
        new.temperature = self.temperature.expand(batch_shape)
        new.logits = self.logits.expand(batch_shape)
        return super().expand(batch_shape, _instance=new)

    def rsample(self, sample_shape=torch.Size(), apply_sigmoid=False):
        shape = self._extended_shape(sample_shape)
        
        dtype = self.temperature.dtype
        # Convert int32/int64 to float32/float64 if necessary
        if dtype in [torch.int32, torch.int64]:
            dtype = torch.float32 if dtype == torch.int32 else torch.float64
        base_dist = torch.rand(shape, dtype=dtype, device=self.temperature.device)
        L = torch.log(base_dist) - torch.log1p(-base_dist)
        if apply_sigmoid:
            y = torch.sigmoid((L + self.logits) / self.temperature)
        else:
            y = (L + self.logits) / self.temperature
        return y

    def sample(self, sample_shape=torch.Size(), return_base_samples=False):
        with torch.no_grad():
            return self.rsample(sample_shape=sample_shape, apply_sigmoid=True)

    def cdf(self, value):
        pass
        #return kumaraswamy_stable_cdf(value, self.log_concentration1, self.log_concentration0)
    
    def log_prob(self, y, with_sigmoid=False):
        r"""
        Compute the log-probability of a given value.

        Args:
            y (torch.Tensor): The value for which to compute the log-probability.

        Returns:
            torch.Tensor: The log-probability of the input value.
        """
        t = self.temperature
        if not with_sigmoid:
            z = self.logits - t * y
            return torch.log(t) + z - 2 * torch.nn.functional.softplus(z)
        elif with_sigmoid:
            logits_y = torch.log(y) - torch.log1p(-y)
            z = self.logits - t * logits_y
            result = torch.where(
                z > 0,
                -self.logits + torch.xlogy(t - 1., y) - (t + 1.) * torch.log1p(-y) - 2 * torch.nn.functional.softplus(-z),
                self.logits - (t + 1.) * torch.log(y) + (t-1.)*torch.log1p(-y) - 2 * torch.nn.functional.softplus(z)) + torch.log(t)
        return result

    @property
    def batch_shape(self):
        return self.mu.size()
    
    @property
    def event_shape(self):
        return torch.Size()
    
    def entropy_estimate(self, num_samples) -> torch.Tensor:
        # Directly estimate the entropy of the distribution using Monte Carlo sampling.
        # H(q) = E_q[-log q(x)] = -E_q[log q(x)]
        value = self.rsample(sample_shape=torch.Size([num_samples]))
        log_probs = self.log_prob(value) # internally clamps
        entropy_estimate = -log_probs.mean(dim=0)
        return entropy_estimate

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # Set up the x-values in the range (low, high)
    low, high = 0, 1
    dtype = torch.float32
    x_values = torch.linspace(low + 1e-8, high - 1e-8, 1000, dtype=dtype)  # Avoid endpoints for numerical instability

    mu = torch.tensor([0.0, 0.0, 4, -4], dtype=dtype)
    log_stdv = torch.tensor([-1, 0.1, 0, 0], dtype=dtype)
    dist = TanhNormal(mu=mu, log_stdv=log_stdv, low=low, high=high)
    log_probs = dist.log_prob(x_values.unsqueeze(-1))
    entropy_samples = 10000

    # turn on gradient tracking in dist
    mu.requires_grad = True
    log_stdv.requires_grad = True

    diff_entropy_estimates = -dist.entropy_estimate(entropy_samples)
    diff_entropy_estimates.backward(torch.ones_like(diff_entropy_estimates))
    grad_mu_fancy = mu.grad.clone()
    grad_log_stdv_fancy = log_stdv.grad.clone()

    samples = dist.sample(sample_shape=torch.Size([entropy_samples])).numpy()

    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    for i, mu_val in enumerate(mu):
        pdf_values = log_probs[:, i].exp().detach().numpy()
        pdf_integral = np.trapz(pdf_values, x_values.numpy())
        # pdf
        axs[0].plot(x_values, pdf_values, label=f'mu={mu_val.item():.2f}, log_stdv={log_stdv[i].item():.2f}, sum={pdf_integral:.2f}, entr={diff_entropy_estimates[i].detach().numpy():.2f}')
        # histogram of samples
        axs[1].hist(samples[:, i], bins=100, density=True, alpha=0.5, label=f'mu={mu_val.item()}, log_stdv={log_stdv[i].item()}')

    axs[0].legend()

    # if ylim > 20, set to 20
    ylim = axs[0].get_ylim()
    if ylim[1] > 20:
        axs[0].set_ylim(0, 20)
    axs[0].set_ylim(0, 3)

    plt.show()