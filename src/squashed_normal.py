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



class TanhNormalLogPDFFromSamples(torch.autograd.Function):
    # to be used to compute log_prob when we have access to eps: y = \tanh(x), x = \mu + \sigma * eps
    @staticmethod
    def forward(ctx, mu: torch.Tensor, log_stdv: torch.Tensor, std_normal_samples: torch.Tensor, scale: float):
        # x = value_inverse = g^{-1}(y) = mu + exp(log_stdv) + eps, where eps \sim N(0, 1)
        # y = value =  g(x) = loc + scale * tanh(x)
        
        value_inverse = mu + torch.exp(log_stdv) * std_normal_samples
        
        # base distribution N(value_inverse; mu, sigma)
        log_pdf_base_contrib = - 0.5 * math.log(2 * math.pi) - log_stdv - 0.5 * ( (value_inverse - mu) / (log_stdv.exp() + 1e-8) ) ** 2
        
        # log abs det jacobian contribution
        log_abs_det_jacob_contrib = - math.log(scale) - safe_log_one_minus_tanh_sq(value_inverse) #- torch.log1p(-torch.tanh(x_inv)**2 + 1e-8)
        
        log_pdf = log_pdf_base_contrib + log_abs_det_jacob_contrib

        # gradients
        grad_mean_contrib = 2 * torch.tanh(value_inverse)
        grad_log_stdv_contrib = 2 * torch.tanh(value_inverse) * log_stdv.exp() * value_inverse * std_normal_samples
        ctx.save_for_backward(grad_mean_contrib, grad_log_stdv_contrib)
  
        return log_pdf

    @staticmethod
    def backward(ctx, grad_output):
        grad_mean_contrib, grad_log_stdv_contrib = ctx.saved_tensors

        grad_mean = grad_output * grad_mean_contrib
        grad_log_stdv = grad_output * grad_log_stdv_contrib
        
        return grad_mean, grad_log_stdv, None, None, None

TanhNormalLogPDF = TanhNormalLogPDFFromSamples.apply









class TanhNormalEntropyEstimate(torch.autograd.Function):
    """
        Use the property of random variable transformation:
        Let X be a random variable and Y = g(X), where g is a strictly monotonic bijection.
        Then H(Y) = H(X) + E[log |g'(X)|].

        Here, y = g(x) = loc + scale * tanh(mu + exp(log_stdv) * x), and x \sim N(0, 1).
        Then g'(x) = scale * (1 - tanh^2(mu + exp(log_stdv) * x)) * exp(log_stdv)

        Recall log(1 - tanh^2(z)) = 2 * (log(2) - z - softplus(-2 * z))
        Then
        u := mu + exp(log_stdv) * x
        g'(x) = scale * exp(log_stdv) * exp[2 * (log(2) - u - softplus(-2 * u)], where u := mu + exp(log_stdv) * x
        log g'(x) = log(scale) + log_stdv + 2 * (log(2) - u - softplus(-2 * u))

        Thus 
        H(Y) 
        =        H(X) + E[log |g'(X)|]
        =        H(X) + log(scale) + log_stdv + 2 * log(2) - 2 * mu - 2 * E[softplus(-2 * u)] <-- estimate this with samples
    """
    @staticmethod
    def forward(ctx, mu: torch.Tensor, log_stdv: torch.Tensor, scale: float, num_samples: torch.Tensor = torch.tensor(10000)):

        standard_normal_contrib = 0.5 * math.log(2 * math.pi * math.e) # x ~ N(0, 1), thus stdv = 1

        ## log_abs_det_jacobian has two components. One can be computed analytically. The other requires sampling.
        # analytical component
        log_abs_det_jacobian_analytic_contrib = math.log(scale) + log_stdv + 2 * math.log(2) - 2 * mu
        
        # sampling component
        shape = torch.Size([num_samples]) + mu.size()
        x_samples = torch.randn(shape, dtype=dtype, device=mu.device)
        u_samples = mu + log_stdv.exp() * x_samples
        log_abs_det_jacobian_estimate_contrib = -2 * torch.nn.functional.softplus(-2 * u_samples).mean(dim=0)

        entropy = standard_normal_contrib + log_abs_det_jacobian_analytic_contrib + log_abs_det_jacobian_estimate_contrib

        # compute gradients here for now
        grad_mu =                          - 2 * torch.mean( torch.tanh(u_samples) ,             dim=0)
        grad_log_stdv = 1 - 2 * log_stdv.exp() * torch.mean( torch.tanh(u_samples) * x_samples , dim=0)

        ctx.save_for_backward(grad_mu, grad_log_stdv)

        return entropy

    @staticmethod
    def backward(ctx, grad_output):
        grad_mu, grad_log_stdv = ctx.saved_tensors
        
        return grad_output * grad_mu, grad_output * grad_log_stdv, None, None

TanhNormalEntropyEstimate = TanhNormalEntropyEstimate.apply




class TanhNormalLogPDF(torch.autograd.Function):
    # to be used for PLOTTING, NOT FOR SAMPLING
    # when we have access to underlying samples y = \tanh(x), x \sim N(mu, \sigma), use TanhNormalLogPDFFromSamples
    @staticmethod
    def forward(ctx, value: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor, eps: torch.Tensor = torch.tensor(1e-8)):
        # eps is a small number to add to sigma to avoid numerical instability

        # inverse_value \in (-1, 1). Cannot be -1 or 1.
        inverse_value = safe_inverse_tanh(value) # x_inv \in (-inf, inf)

        # naive implementation -- works!
        #log_pdf = torch.distributions.Normal(mu, torch.exp(log_sigma)).log_prob(x_inv) - torch.log1p(-torch.tanh(x_inv)**2 + 1e-8)
        log_pdf = torch.distributions.Normal(mu, torch.exp(log_sigma)).log_prob(inverse_value) - safe_log_one_minus_tanh_sq(inverse_value + 1e-8)
        # more stable implementation -- works!
        #log_pdf = torch.distributions.Normal(mu, torch.exp(log_sigma)).log_prob(x_inv) - 2 * (math.log(2) - x_inv - torch.nn.functional.softplus(-2 * x_inv))

        # without torch.distributions.Normal -- works!
        #scaled_squared_error = ( (x_inv - mu) / (torch.exp(log_sigma) + eps) ) ** 2
        #normal_contrib = - 0.5 * math.log(2 * math.pi) - log_sigma - scaled_squared_error
        #log_abs_det_jacobian_contrib = - 2 * (math.log(2) - x_inv - torch.nn.functional.softplus(-2 * x_inv))
        #log_pdf = normal_contrib + log_abs_det_jacobian_contrib
        ctx.save_for_backward(inverse_value, mu, log_sigma, eps)

        return log_pdf

    @staticmethod
    def backward(ctx, grad_output):
        inverse_value, mu, log_sigma, eps = ctx.saved_tensors
        standardized_error = (inverse_value - mu) / (torch.exp(2 * log_sigma) + eps)

        grad_x_inv = grad_mean = grad_log_sigma = None

        if ctx.needs_input_grad[0]:
            # - (inverse_value - \mu) / \sigma^2 + 2 * tanh^2(x_inv)
            grad_x_inv_contrib = - standardized_error + 2 * torch.tanh(inverse_value) ** 2
            grad_x_inv = grad_output * grad_x_inv_contrib
        
        if ctx.needs_input_grad[1]:
            grad_mean = grad_output * standardized_error
        
        if ctx.needs_input_grad[2]:
            # - 1 / sigma (1 - \frac{inverse_value - mu)}{sigma}^2 )
            scaled_squared_error = ( (inverse_value - mu) / (torch.exp(log_sigma) + eps) ) ** 2
            inv_sigma = torch.exp(-log_sigma)
            if torch.any(scaled_squared_error <= 0):
                print("WARNING: ( (z-mu)/sigma )**2 has underflowed to 0. Clamping to 2**-23.")
                scaled_squared_error = scaled_squared_error.clamp(min=2**-23) # ensure scaled_squared_error > 0 <-- TODO: eventually replace with log1mexp
            
            grad_log_sigma_contrib = - inv_sigma * (1 - scaled_squared_error)
            grad_log_sigma = grad_output * grad_log_sigma_contrib
        
        return grad_x_inv, grad_mean, grad_log_sigma, None

TanhNormalLogPDF = TanhNormalLogPDF.apply



class TanhNormalSampler(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mu: torch.Tensor, log_std: torch.Tensor, std_normal_samples: torch.Tensor, low: float, high: float):
        loc, scale = (high + low) / 2, (high - low) / 2
        x = mu + torch.exp(log_std) * std_normal_samples
        y = loc + scale * torch.tanh(x) # \in (loc - scale, loc + scale)

        grad_mean_contrib = scale * torch.exp( safe_log_one_minus_tanh_sq(x) )
        grad_log_stdv_contrib = scale * torch.exp( safe_log_one_minus_tanh_sq(x) + log_std) * std_normal_samples
        ctx.save_for_backward(grad_mean_contrib, grad_log_stdv_contrib)
        
        return y

    @staticmethod
    def backward(ctx, grad_output):
        grad_mean_contrib, grad_log_stdv_contrib = ctx.saved_tensors
        
        grad_mean, grad_log_std = None

        if ctx.needs_input_grad[0]:
            grad_mean = grad_output * grad_mean_contrib
        if ctx.needs_input_grad[1]:
            grad_log_std = grad_output * grad_log_stdv_contrib
        
        return grad_mean, grad_log_std, None, None, None
    

TanhNormalSampler = TanhNormalSampler.apply

class TanhNormal(Distribution):
    r"""
    Samples from a TanhNormal distribution with numerically stabilized expressions for log_prob and sample differentiation.
    
    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = TanhNormal(torch.tensor([-1.0]), torch.tensor([0.0]))
        >>> m.sample()  # sample from a TanhNormal distribution with concentration mu=-1 and log_stdv=0
        tensor([ -0.9132])

    Args:
        loc (torch.Tensor): normal distribution location parameter
        log_scale (torch.Tensor): log of the normal distribution stdv parameter (squared root of variance)
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
                 low: float = -1.0, # TODO: make these tensors, allow for differing low/high values
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

        ONLY USE VALUE_INVERSE FOR PLOTTING PURPOSES, will disrupt gradients.

        The `TanhNormalLogPDF` is defined on the canonical domain (-1, 1). To compute
        the log-probability for a value `x` in the interval `[low, high]`, we first apply
        an affine transformation that maps `x` from `[low, high]` back to the `[-1, 1]` 
        domain. Let `T(y)` be the affine transformation that maps `y \in [-1, 1]` to 
        `[low, high]`, i.e.:

        .. math::
            \text{loc} &= \frac{\text{high} + \text{low}}{2} \\
            \text{scale} &= \frac{\text{high} - \text{low}}{2} \\
            T(y) &= \text{loc} + \text{scale} \cdot y \\
            T^{-1}(x) &= \frac{x - \text{loc}}{\text{scale}}

        The derivative of the inverse transformation is:

        .. math::
            \frac{d T^{-1}(x)}{dx} = \frac{1}{\text{scale}} \quad (\text{positive since } \text{high} > \text{low})

        Using the change of variables formula, we compute the log-probability as:

        .. math::
            \log p(x) = \log p(T^{-1}(x)) + \log \left| \frac{d T^{-1}(x)}{dx} \right| 
                    = \log p(T^{-1}(x)) - \log \text{scale}

        Args:
            value (torch.Tensor): The value for which to compute the log-probability.
            value_inverse (torch.Tensor, optional): The inverse of the value through the transformed tanh. 

        Returns:
            torch.Tensor: The log-probability of the input value.
        """
        loc, scale = (self.high + self.low) / 2, (self.high - self.low) / 2

        
        # if value_inverse directly provided, no need to compute it
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
        scale = (self.high - self.low) / 2
        return TanhNormalEntropyEstimate(self.mu, self.log_stdv, scale, num_samples)

    def entropy_estimate_log_pdf(self, num_samples) -> torch.Tensor:
        # estimate the KL between the distribution and a uniform distribution with num_samples samples
        value = self.rsample(sample_shape=torch.Size([num_samples]))
        log_probs = self.log_prob(value) #, value_inverse=value_inv) # internally clamps
        entropy_estimate = -log_probs.mean(dim=0)
        return entropy_estimate
    
    def entropy_estimate_2(self, num_samples) -> torch.Tensor:
        """
        Use the property of random variable transformation:
        Let X be a random variable and Y = g(X), where g is a strictly monotonic bijection.
        Then H(Y) = H(X) + E[log |g'(X)|].

        Here, y = g(x) = loc + scale * tanh(mu + exp(log_stdv) * x), and X \sim N(0, 1).
        Then g'(x) = scale * (1 - tanh^2(mu + exp(log_stdv) * x)) * exp(log_stdv)

        Recall log(1 - tanh^2(z)) = 2 * (log(2) - z - softplus(-2 * z))
        Then
        u := mu + exp(log_stdv) * x
        g'(x) = scale * exp(log_stdv) * exp[2 * (log(2) - u - softplus(-2 * u)], where u := mu + exp(log_stdv) * x
        log g'(x) = log(scale) + log_stdv + 2 * (log(2) - u - softplus(-2 * u))

        Thus 
        H(Y) 
        =        H(X) + E[log |g'(X)|]
        =        H(X) + scale * exp(log_stdv) * exp(2 log(2)) * E[ exp[-2 * (u + softplus(-2 * u))] ]
        \approx  H(X) + scale * exp(log_stdv) * exp(2 log(2)) * [ \sum_i exp[-2 * (u_i + softplus(-2 * u_i))] ] / n
         where u_i = mu + exp(log_stdv) * x_i, x_i \sim N(0, 1)
        
        """
        entropy_normal = 0.5 * math.log(2 * math.pi * math.e) # x ~ N(0, 1), thus stdv = 1
        loc, scale = (self.high + self.low) / 2, (self.high - self.low) / 2
        
        shape = self._extended_shape(torch.Size([num_samples]))
        x_samples = torch.randn(shape, dtype=dtype, device=self.mu.device)
        u_samples = self.mu + self.log_stdv.exp() * x_samples

        estimate_log_g_prime = math.log(scale) + self.log_stdv + safe_log_one_minus_tanh_sq(u_samples).mean(dim=0)

        entropy_estimate = entropy_normal + estimate_log_g_prime
        return entropy_estimate
    """ 
    def entropy_estimate_3(self, num_samples) -> torch.Tensor:
        loc, scale = (self.high + self.low) / 2, (self.high - self.low) / 2
        entropy_normal = 0.5 * math.log(2 * math.pi * math.e) # x ~ N(0, 1), thus stdv = 1
        entropy_log_abs_det_jacobian_analytic_contrib = math.log(scale) + self.log_stdv + 2 * math.log(2) - 2 * self.mu

        shape = self._extended_shape(torch.Size([num_samples]))
        x_samples = torch.randn(shape, dtype=dtype, device=self.mu.device)
        u_samples = self.mu + self.log_stdv.exp() * x_samples
        entropy_log_abs_det_jacobian_estimate_contrib = -2 * torch.nn.functional.softplus(-2 * u_samples).mean(dim=0)
        entropy_estimate = entropy_normal + entropy_log_abs_det_jacobian_analytic_contrib + entropy_log_abs_det_jacobian_estimate_contrib
        return entropy_estimate
    """

# define main function so i can run this script
if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # Set up the x-values in the range (low, high)
    low, high = 0, 1
    dtype = torch.float32
    x_values = torch.linspace(low + 1e-8, high - 1e-8, 1000, dtype=dtype)  # Avoid endpoints for numerical instability


    #mu = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0, 2.0], dtype=dtype)
    #log_stdv = torch.tensor([-1.0, -0.5, -0.25, -0.0, 0.1, 0.5], dtype=dtype)
    #entropy_samples = 10000000

    mu = torch.tensor([0.0, 0.0, 4, -4], dtype=dtype)
    log_stdv = torch.tensor([-1, 0.1, 0, 0], dtype=dtype)
    dist = TanhNormal(mu=mu, log_stdv=log_stdv, low=low, high=high)
    log_probs = dist.log_prob(x_values.unsqueeze(-1))
    entropy_samples = 10000
    # extract gradient of entropy wrt mu and log_stdv
    #diff_entropy_estimates = - dist.entropy_estimate(entropy_samples)
    #diff_entropy_estimates_log_pdf = - dist.entropy_estimate_log_pdf(entropy_samples)

    # turn on gradient tracking in dist
    mu.requires_grad = True
    log_stdv.requires_grad = True

    diff_entropy_estimates = -dist.entropy_estimate(entropy_samples)
    diff_entropy_estimates.backward(torch.ones_like(diff_entropy_estimates))
    grad_mu_fancy = mu.grad.clone()
    grad_log_stdv_fancy = log_stdv.grad.clone()

    # Zero the gradients to compute for the second method
    mu.grad.zero_()
    log_stdv.grad.zero_()

    # Compute the gradients for the log-prob based method
    diff_entropy_estimates_log_pdf = -dist.entropy_estimate_log_pdf(entropy_samples)
    diff_entropy_estimates_log_pdf.backward(torch.ones_like(diff_entropy_estimates_log_pdf))
    grad_mu_log_pdf = mu.grad.clone()
    grad_log_stdv_log_pdf = log_stdv.grad.clone()

    # Print the gradients for comparison
    print("Gradient comparison:")
    print("MC-based entropy estimation (mu gradients):", grad_mu_fancy)
    print("Log-prob-based entropy estimation (mu gradients):", grad_mu_log_pdf)
    print("MC-based entropy estimation (log_stdv gradients):", grad_log_stdv_fancy)
    print("Log-prob-based entropy estimation (log_stdv gradients):", grad_log_stdv_log_pdf)

    samples = dist.sample(sample_shape=torch.Size([entropy_samples])).numpy()

    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    for i, mu_val in enumerate(mu):
        pdf_values = log_probs[:, i].exp().detach().numpy()
        print(f"Difference in entropy estimates (naive vs fancy): {diff_entropy_estimates[i].detach().numpy() - diff_entropy_estimates_log_pdf[i].detach().numpy()}")
        #print(f"Difference in entropy estimates (naive vs 3): {entropy_estimate - entropy_estimate3}")
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


    """ 
    # Parameters to test
    loc, scale = (high + low) / 2, (high - low) / 2
    mu_values = [2]  # Example values for mu
    log_stdv_values = [-3, -1, -.25, 0, 2]  # Example values for log_sigma
    def compute_log_pdf(x_values, mu, log_sigma, x_inv=None):
        x_inv = safe_inverse_tanh( (x_values - loc) / scale  ) if x_inv is None else x_inv
        log_pdf_base_contrib = - 0.5 * math.log(2 * math.pi) - log_sigma - 0.5 * ( (x_inv - mu) / (torch.exp(log_sigma) + 1e-8) ) ** 2
        #log_pdf = - math.log(scale) + log_pdf_base_contrib - torch.log1p(-torch.tanh(x_inv)**2 + 1e-8)
        log_pdf = - math.log(scale) + log_pdf_base_contrib - safe_log_one_minus_tanh_sq(x_inv)
        log_pdf = torch.clamp(log_pdf, min = math.log10(EPS)) # <- **CLAMPING THIS SEEMS TO RESOLVE THE ISSUE**

        #log_pdf = - math.log(scale) + torch.distributions.Normal(mu, torch.exp(log_sigma)).log_prob(x_inv) - torch.log1p(-torch.tanh(x_inv)**2 + 1e-8)
        #log_pdf = - math.log(scale) + torch.distributions.Normal(mu, torch.exp(log_sigma)).log_prob(x_inv) - safe_log_one_minus_tanh_sq(x_inv)
        # print max diff
        #print(torch.max(log_pdf - log_pdf2))
        return log_pdf


    def estimate_entropy(mu, log_sigma, num_samples):
        # estimate the KL between the distribution and a uniform distribution with num_samples samples
        z = mu + torch.exp(log_sigma) * torch.randn(num_samples)
        y = loc + scale * torch.tanh(z)
        log_probs = compute_log_pdf(y, mu, log_sigma, x_inv=z)
        entropy_estimate = -log_probs.mean()
        return entropy_estimate
    """