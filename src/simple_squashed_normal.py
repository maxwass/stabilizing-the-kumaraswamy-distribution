import math
import torch
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all
import numpy as np

def safe_inverse_tanh(y):
    # from https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/distributions.html#TanhBijector
    
    # Clip y to avoid NaN
    eps = torch.finfo(y.dtype).eps
    y = y.clamp(min=-1.0 + eps, max=1.0 - eps)
    return 0.5 * (y.log1p() - (-y).log1p())

def safe_log_one_minus_tanh_sq(x):
    '''
    From PyTorch `TanhTransform`
    log(1-tanh^2(x)) 
    = log(sech^2(x)) 
    = 2log(2/(e^x + e^(-x))) 
    = 2(log2 - log(e^x/(1 + e^(-2x)))
    = 2(log2 - x - log(1 + e^(-2x)))
    = 2(log2 - x - softplus(-2x)) 
    '''
    return 2 * (math.log(2) - x - torch.nn.functional.softplus(-2 * x))


# squashed normal implementation with all basic fixes, namely
# 1. safe inverse tanh
# 2. safe log(1 - tanh^2(x)) via the softplus trick
# 3. clamping the log_prob to avoid numerical instability

class TanhNormal(Distribution):
    r"""
    Squashed Normal distribution through a tanh transformation.
    
    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = TanhNormal(torch.tensor([-1.0]), torch.tensor([0.0]))
        >>> m.sample()  # sample from a TanhNormal distribution with concentration mu=-1 and log_stdv=0
        tensor([ -0.9132])

    Args:
        mu (torch.Tensor): normal distribution location parameter
        log_stdv (torch.Tensor): log of the normal distribution stdv parameter (squared root of variance)
        low (torch.Tensor or number, optional): minimum value of the distribution. Default is -1.0;
        high (torch.Tensor or number, optional): maximum value of the distribution. Default is 1.0;
    """
    arg_constraints = {
        "mu": constraints.real,
        "log_stdv": constraints.real, # for now, leave unconstrained below
        "low": constraints.real,
        "high": constraints.real,
    }
    #support = constraints.interval(-1, 1)
    has_rsample = True
    
    def __init__(self, 
                 mu: torch.Tensor, 
                 log_stdv: torch.Tensor, 
                 low: float = -1.0,
                 high: float = 1.0,
                 log_prob_clamp_lower: float = math.log10(1e-7),
                 log_prob_clamp_upper: float = math.inf,
                 validate_args=None):
        #self.log_stdv = broadcast_all(torch.as_tensor(mu), torch.as_tensor(log_stdv))
        self.mu, self.log_stdv = broadcast_all(mu, log_stdv)
        
        # We do NOT subclass TransformedDistribution. We manually implement the rsample method.

        device = self.mu.device
        
        err_msg = "TanhNormal: high must be strictly greater than low values {} < {}".format(low, high)
        if low >= high:
            raise RuntimeError(err_msg)
        self.low = torch.as_tensor(low, device=device)
        self.high = torch.as_tensor(high, device=device)

        self.log_prob_clamp_lower = torch.tensor(log_prob_clamp_lower, device=device)
        self.log_prob_clamp_upper = torch.tensor(log_prob_clamp_upper, device=device)


        self.batch_shape_ = self.mu.size() # batch_shape = self.mu.size()
        super().__init__(batch_shape=self.mu.size(), validate_args=validate_args) #validate_args=validate_args)
    
    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(TanhNormal, _instance)
        new.mu = self.mu.expand(batch_shape)
        new.log_stdv = self.log_stdv.expand(batch_shape)
        return super().expand(batch_shape, _instance=new)

    def rsample(self, sample_shape=torch.Size(), return_base_samples=False):
        """
        Explicit reparameterized sampling from the TanhNormal distribution.

        Args:
            sample_shape (torch.Size, optional): The shape of the samples.

        Returns:
            torch.Tensor: Sampled tensor
        """
        
        shape = self._extended_shape(sample_shape)
        
        dtype = self.mu.dtype
        # Convert int32/int64 to float32/float64 if necessary
        if dtype in [torch.int32, torch.int64]:
            dtype = torch.float32 if dtype == torch.int32 else torch.float64

        loc, scale = (self.high + self.low) / 2, (self.high - self.low) / 2
        base_dist = torch.randn(shape, dtype=dtype, device=self.mu.device)
        z = self.mu + self.log_stdv.exp() * base_dist
        y = loc + scale * torch.tanh(z)

        if return_base_samples:
            return y, z
        else:
            return y

    def sample(self, sample_shape=torch.Size(), return_base_samples=False):
        with torch.no_grad():
            return self.rsample(sample_shape=sample_shape, return_base_samples=return_base_samples)

    def cdf(self, value):
        pass
        #return kumaraswamy_stable_cdf(value, self.log_concentration1, self.log_concentration0)
    
    def log_prob(self, value, value_inverse=None):
        r"""
        Compute the log-probability of a given value in the [low, high] interval.

        Args:
            value (torch.Tensor): The value for which to compute the log-probability.
            value_inverse (torch.Tensor, optional): The inverse of the value through the transformed tanh. 

        Returns:
            torch.Tensor: The log-probability of the input value.
        """
        loc, scale = (self.high + self.low) / 2, (self.high - self.low) / 2

        # if value_inverse directly provided, no need to compute it, but will not drive gradients through value
        value_inverse = safe_inverse_tanh( (value - loc) / scale  ) if value_inverse is None else value_inverse
        
        mu, log_stdv = self.mu, self.log_stdv
        #mu, log_stdv, value, value_inverse = broadcast_all(self.mu, self.log_stdv, value, value_inverse)

        # base distribution N(value_inverse; mu, sigma)
        log_pdf_base_contrib = - 0.5 * math.log(2 * math.pi) - log_stdv \
            - 0.5 * ( (value_inverse - mu) / (log_stdv.exp() + 1e-8) ) ** 2
        
        # log abs det jacobian contribution
        log_abs_det_jacob_contrib = - math.log(scale) - safe_log_one_minus_tanh_sq(value_inverse) #- torch.log1p(-torch.tanh(x_inv)**2 + 1e-8)
        
        log_pdf = log_pdf_base_contrib + log_abs_det_jacob_contrib
        
        log_pdf = torch.clamp(log_pdf, min=self.log_prob_clamp_lower, max=self.log_prob_clamp_upper) # <- ** CRITICAL STEP: CLAMPING THIS SEEMS TO RESOLVE ISSUES**

        return log_pdf

    @property
    def batch_shape(self):
        return self.mu.size()
    
    @property
    def event_shape(self):
        return torch.Size()

    @property
    def mean(self):
        pass # no analytical solution for moment of tanh normal
    
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