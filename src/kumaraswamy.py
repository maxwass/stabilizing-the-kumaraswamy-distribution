import math
import torch
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all
from torch import exp, log
from typing import Optional

class KumaraswamyStableLogPDF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, log_a: torch.Tensor, log_b: torch.Tensor, max_grad_log_a_clamp: Optional[float] = None):
        # x \in (0, 1). Cannot be 0 or 1.
        log_x = log(x)
        alogx = exp(log_a) * log_x
        z = log1mexp(- alogx)
        am1 = torch.expm1(log_a)
        bm1 = torch.expm1(log_b)
        log_pdf = log_a + log_b + am1 * log_x + bm1 * z

        # To save for backward, must be tensors, not floats
        max_grad_log_a_clamp = torch.tensor(max_grad_log_a_clamp) if max_grad_log_a_clamp is not None else None
        
        ctx.save_for_backward(log_a, log_b, log_x, am1, bm1, alogx, z, x, max_grad_log_a_clamp)

        return log_pdf

    @staticmethod
    def backward(ctx, grad_output):
        log_a, log_b, log_x, am1, bm1, alogx, z, x, max_grad_log_a_clamp = ctx.saved_tensors
        
        grad_x = grad_log_a = grad_log_b = None

        if ctx.needs_input_grad[0]:
            # grad_x_contrib = am1 / x  - bm1 * exp(log_a + am1 * log_x - z)
            
            # HAZARD: The largest expressible number in float32 is ~3.41e38. Then if x below 1e-38, 1/x is 
            # above 1e38, producing inf. Simple fix: clamp 1/x to large representable number.
            x_inv = (1 / x).clamp(max=1e12) # TODO: adjust for single vs double precision

            # HAZARD: The largest expressible number in float32 is ~3.41e38. if exp_arg is above 87, exp(exp_arg) is inf
            exp_arg = torch.clamp(log_a + am1 * log_x - z, max=87.0) # TODO: adjust for single vs double precision
            
            grad_x_contrib = am1 * x_inv - bm1 * exp(exp_arg)
            grad_x = grad_output * grad_x_contrib
        
        if ctx.needs_input_grad[1]:
            b = exp(log_b)
            grad_log_a_contrib = 1 + alogx * (1 - bm1 * exp(alogx - z))
            
            # For training on data with many observations \approx 0 or 1 (MNIST), often \nabla_{log_a} become very positive. 
            # Observe more stable training with clipping.
            if max_grad_log_a_clamp is not None:
                grad_log_a_contrib = grad_log_a_contrib.clamp(max=.2) # TODO: change this from 0.2 to max_grad_log_a_clamp
 
            grad_log_a = grad_output * grad_log_a_contrib
        
        if ctx.needs_input_grad[2]:
            
            b = exp(log_b)
            nabla_log_b = 1 + b * z
            grad_log_b = grad_output * nabla_log_b
        
        return grad_x, grad_log_a, grad_log_b, None


kumaraswamy_stable_log_pdf = KumaraswamyStableLogPDF.apply


class KumaraswamyStableLogPDFWithLogxInput(torch.autograd.Function):
    """
        Not currently used. Purpose: with access to log(x) as input (vs x), we have perform more accurate/stable computation.
    """
    @staticmethod
    def forward(ctx, log_x: torch.Tensor, log_a: torch.Tensor, log_b: torch.Tensor, max_grad_log_a_clamp: Optional[float] = None):
        # x \in (0, 1). Cannot be 0 or 1. --> log_x \in (-inf, 0)
        assert torch.all(log_x < 0), "log_x must be negative"
        assert torch.all(log_x > -np.inf), "log_x must be finite" # TODO: remove np

        alogx = exp(log_a) * log_x
        z = log1mexp(- alogx)
        am1 = torch.expm1(log_a)
        bm1 = torch.expm1(log_b)
        log_pdf = log_a + log_b + am1 * log_x + bm1 * z

        # To save for backward, must be tensors, not floats
        max_grad_log_a_clamp = torch.tensor(max_grad_log_a_clamp) if max_grad_log_a_clamp is not None else None

        ctx.save_for_backward(log_a, log_b, am1, bm1, alogx, z, max_grad_log_a_clamp)

        return log_pdf

    @staticmethod
    def backward(ctx, grad_output):
        log_a, log_b, am1, bm1, alogx, z, max_grad_log_a_clamp = ctx.saved_tensors
        
        grad_log_x = grad_log_a = grad_log_b = None

        if ctx.needs_input_grad[0]:
            exp_arg = torch.clamp(alogx - z + log_a, max=100.0) # ensures exp is finite. TODO: adjust for single vs double precision
            grad_log_x_contrib = am1 - bm1 * exp(exp_arg)
            grad_log_x = grad_output * grad_log_x_contrib
        if ctx.needs_input_grad[1]:
            grad_log_a_contrib = 1 + alogx * (1 - bm1 * exp(alogx - z))
            grad_log_a = grad_output * grad_log_a_contrib
        if ctx.needs_input_grad[2]:
            b = exp(log_b)
            grad_log_b_contrib = 1 + b * z
            grad_log_b = grad_output * grad_log_b_contrib
        
        return grad_log_x, grad_log_a, grad_log_b, None


kumaraswamy_stable_log_pdf_log_x_input = KumaraswamyStableLogPDFWithLogxInput.apply


class KumaraswamyStableInverseCDF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u: torch.Tensor, log_a: torch.Tensor, log_b: torch.Tensor):
        alpha, beta = exp(- log_a), exp(- log_b) # 1/a, 1/b
        
        dtype, device = u.dtype, u.device
        # in single precision, smallest_subnormal approx exp(-103)
        smallest_subnormal = torch.tensor(1.401298464324817e-45, dtype=dtype, device=device) if dtype == torch.float32 else torch.tensor(4.9406564584124654e-324, dtype=dtype, device=device)
        largest_number_less_than_one = torch.tensor(1 - 2**-24, dtype=dtype, device=device) if dtype == torch.float32 else torch.tensor(1 - 2**-53, dtype=dtype, device=device)
        u = u.clamp(min=smallest_subnormal)

        log_u = log(u)
        c = beta * log_u
        c = c.clamp(max = - smallest_subnormal) # enforce c < 0
        z = log1mexp(- c)
        F_inv = exp(alpha * z)

        # Output must be in (0, 1), not [0, 1], and exp can push some very small values of alpha * z to 1
        F_inv = F_inv.clamp(min=smallest_subnormal, max=largest_number_less_than_one)

        ctx.save_for_backward(u, log_u, alpha, log_a, log_b, c, z)
        
        return F_inv

    @staticmethod
    def backward(ctx, grad_output):
        u, log_u, alpha, log_a, log_b, c, z = ctx.saved_tensors

        # TODO: optionally clamp exp arguments in grad_log_a_contrib & grad_log_b_contrib to prevent overflow/underflow
        grad_u = grad_log_a = grad_log_b = None

        if ctx.needs_input_grad[0]:
            grad_u = torch.zeros_like(u, device=grad_output.device)
        if ctx.needs_input_grad[1]:
            grad_log_a_contrib = exp(- log_a + alpha * z) * (-z)
            grad_log_a = grad_output * grad_log_a_contrib
        if ctx.needs_input_grad[2]:
            grad_log_b_contrib = exp(- log_a - log_b + c + torch.expm1(- log_a) * z) * log_u
            grad_log_b = grad_output * grad_log_b_contrib
        
        return grad_u, grad_log_a, grad_log_b
    

kumaraswamy_stable_inverse_cdf = KumaraswamyStableInverseCDF.apply


class KumaraswamyStableLogInverseCDF(torch.autograd.Function):
    """
        Not currently used. Purpose: with access to log(x) as input (vs x), we have perform more accurate/stable computation.
    """
    @staticmethod
    def forward(ctx, u: torch.Tensor, log_a: torch.Tensor, log_b: torch.Tensor):
        """
        Computes the log of the inverse CDF of the Kumaraswamy distribution. Thus outputs are in (-inf, 0).
        This can be used for potentially more stable training than `KumaraswamyStableInverseCDF` when log probabilities are desired.
        These are NOT logits. They CANNOT be used as inputs to, e.g., BCEWithLogitsLoss. They are log probabilities.
        """

        alpha, beta = exp(- log_a), exp(- log_b) # 1/a, 1/b
        
        dtype, device = u.dtype, u.device
        smallest_subnormal = torch.tensor(1.401298464324817e-45, dtype=dtype, device=device) if dtype == torch.float32 else torch.tensor(4.9406564584124654e-324, dtype=dtype, device=device)
        u = u.clamp(min=smallest_subnormal)

        log_u = log(u)
        c = beta * log_u # c < 0
        
        # enforce c < 0
        c_underflow_mask = (c >= 0)
        if False and c_underflow_mask.any():
            print("** Underflow in KumaraswamyStableLogInverseCDF. Setting c to -smallest_subnormal")
            print(f"\tu = {u}\n\tlog_a = {log_a}\n\tlog_b = {log_b}\n\tc = {c}")
        c[c_underflow_mask] = -smallest_subnormal
        
        z = log1mexp(- c)
        log_F_inv = alpha * z

        # Ensure the result remains well-behaved (finite and not exactly 0 or 1) in single precision when used inside an exponent.
        #  TODO: adjust for single vs double precision
        log_F_inv = log_F_inv.clamp(min=-100, # in single precision above ~103  will cause exp(log_F_inv) to underflow to 0
                                    max=-2**-24) # similarly, above this will cause exp(log_F_inv) to be exactly 1

        ctx.save_for_backward(u, log_u, log_a, log_b, c, z, log_F_inv)
        
        return log_F_inv

    @staticmethod
    def backward(ctx, grad_output):
        u, log_u, log_a, log_b, c, z, log_F_inv = ctx.saved_tensors

        grad_u = grad_log_a = grad_log_b = None

        if ctx.needs_input_grad[0]:
            grad_u = torch.zeros_like(u, device=grad_output.device)  # Assuming gradients w.r.t. u are not needed
        if ctx.needs_input_grad[1]:
            grad_log_a_contrib = - log_F_inv
            grad_log_a = grad_output * grad_log_a_contrib
        if ctx.needs_input_grad[2]:
            # TODO: adjust for single vs double precision
            exp_arg = torch.clamp(-log_a - log_b + c - z, max=87.0) # in single precision above ~87.722 will cause exp(x) to overflow to inf
            grad_log_b_contrib = exp(exp_arg) * log_u
            grad_log_b = grad_output * grad_log_b_contrib
        
        return grad_u, grad_log_a, grad_log_b


kumaraswamy_stable_log_inverse_cdf = KumaraswamyStableLogInverseCDF.apply


class KumaraswamyStableCDF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, log_a: torch.Tensor, log_b: torch.Tensor):
        a = exp(log_a)
        b = exp(log_b)
        log_x = log(x)
        c = a * log_x
        z = log1mexp(- c)
        cdf = -torch.expm1(b * z)
        ctx.save_for_backward(log_a, log_b, log_x, a, b, c, z)
        return cdf
    
    @staticmethod
    def backward(ctx, grad_output):
        # Note: this is nabla_log_a NOT nabla_a. We take as input log_a and log_b, not a and b. 
        pass


kumaraswamy_stable_cdf = KumaraswamyStableCDF.apply




class KumaraswamyStable(Distribution):
    r"""
    Samples from a Kumaraswamy distribution with numerically stabilized expressions and unconstrained parameterization.
    
    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = KumaraswamyStable(torch.tensor([-1.0]), torch.tensor([1.0]))
        >>> m.sample()  # sample from a Kumaraswamy distribution with concentration log_a=-1 and log_b=1
        tensor([ 0.01321])

    Args:
        log_concentration1 (float or Tensor): log of the 1st concentration parameter of the distribution
            (often referred to as a or alpha)
        log_concentration0 (float or Tensor): log of the 2nd concentration parameter of the distribution
            (often referred to as b or beta)
    """
    arg_constraints = {
        "log_concentration1": constraints.real, # log_a
        "log_concentration0": constraints.real, # log_b
    }
    support = constraints.unit_interval
    has_rsample = True
    
    def __init__(self, log_concentration1, log_concentration0, validate_args=None):
        self.log_concentration1, self.log_concentration0 = broadcast_all(
            log_concentration1, log_concentration0)
        
        # We do NOT subclass TransformedDistribution. We manually implement the rsample method.
        
        batch_shape = self.log_concentration1.size()
        super().__init__(batch_shape, validate_args=validate_args)
    
    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(KumaraswamyStable, _instance)
        new.log_concentration1 = self.log_concentration1.expand(batch_shape)
        new.log_concentration0 = self.log_concentration0.expand(batch_shape)
        return super().expand(batch_shape, _instance=new)

    def rsample(self, sample_shape=torch.Size(), log_sample: bool = False):
        """
        Explicit reparameterized sampling from the Kumaraswamy distribution.

        Args:
            sample_shape (torch.Size, optional): The shape of the samples.
            log_sample (bool, optional): If True, returns the log of the sample, useful for numerical stability
                when passing log probabilities to functions such as binary cross-entropy loss.

        Returns:
            torch.Tensor: Sampled tensor, either the sample itself or its log value depending on `log_sample`.
        """

        shape = self._extended_shape(sample_shape)
        dtype = self.log_concentration1.dtype
        
        # Convert int32/int64 to float32/float64 if necessary
        if dtype in [torch.int32, torch.int64]:
            dtype = torch.float32 if dtype == torch.int32 else torch.float64
        
        base_dist = torch.rand(shape, dtype=dtype, device=self.log_concentration1.device)
        
        if log_sample:
            return kumaraswamy_stable_log_inverse_cdf(base_dist, self.log_concentration1, self.log_concentration0) # not currently used
        else:
            return kumaraswamy_stable_inverse_cdf(base_dist, self.log_concentration1, self.log_concentration0)

    def sample(self, sample_shape=torch.Size(), log_sample: bool = False):
        with torch.no_grad():
            return self.rsample(sample_shape=sample_shape, log_sample=log_sample)

    def cdf(self, value):
        return kumaraswamy_stable_cdf(value, self.log_concentration1, self.log_concentration0)

    def log_prob(self, value, max_grad_log_a_clamp: Optional[float] = None, log_value: bool = False):
        """
        Compute the log-probability of a given value.

        Args:
            value (torch.Tensor): The value for which to compute the log-probability.
            max_grad_log_a_clamp (Optional[float], optional): Max clamp value for gradient of log_a.
            log_value (bool, optional): If True, treats the input as `log(value)` instead of `value`.

        Returns:
            torch.Tensor: The log-probability of the input value.
        """
        if log_value:
            return kumaraswamy_stable_log_pdf_log_x_input(value, self.log_concentration1, self.log_concentration0, max_grad_log_a_clamp) # not currently used
        else:
            return kumaraswamy_stable_log_pdf(value, self.log_concentration1, self.log_concentration0, max_grad_log_a_clamp)

    @property
    def batch_shape(self):
        return self.log_concentration1.size()
    
    @property
    def event_shape(self):
        return torch.Size()

    @property
    def mean(self):
        return _moments(torch.exp(self.log_concentration1), torch.exp(self.log_concentration0), 1)
    
    def entropy(self) -> torch.Tensor:
        # for now use default pytorch entropy
        return torch.distributions.Kumaraswamy(torch.exp(self.log_concentration1), torch.exp(self.log_concentration0)).entropy()
    
#class KumaraswamyStablePyro(KumaraswamyStable, TorchDistributionMixin):
#    pass


### Utils

def _moments(a, b, n):
    """
    Computes nth moment of Kumaraswamy using using torch.lgamma

    Implementation from pytorch distributions Kumaraswamy

    """
    arg1 = 1 + n / a
    log_value = torch.lgamma(arg1) + torch.lgamma(b) - torch.lgamma(arg1 + b)
    return b * torch.exp(log_value)


def smallest_subnormal_value(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """
    Returns the smallest subnormal value for the given data type.

    Args:
        dtype (torch.dtype): The data type, either torch.float32 or torch.float64.
        device (torch.device): The device to place the tensor on.

    Returns:
        torch.Tensor: The smallest subnormal value for the specified dtype.
    """
    if dtype == torch.float32:
        return torch.tensor(1.401298464324817e-45, dtype=dtype, device=device)
    else:
        return torch.tensor(4.9406564584124654e-324, dtype=dtype, device=device)


# pytorch log1mexp
# https://github.com/pytorch/pytorch/issues/39242 # I THINK THIS HAS A BUG: no negative before exp in second case

def log1mexp(a: torch.Tensor) -> torch.Tensor:
    """
    Computes log(1 - exp(-a)) in a numerically stable way for positive `a`.

    Args:
        a (torch.Tensor): A positive tensor.

    Returns:
        torch.Tensor: The result of log(1 - exp(-a)), computed stably.
    """
    mask = a < math.log(2)
    return torch.where(
        mask,
        torch.log(-torch.expm1(-a)),
        torch.log1p(-torch.exp(-a)),
    )


def kl_divergence_estimator(q, p, num_samples):
    """
    Approximates the KL divergence between torch.distributions q and p

    See http://joschu.net/blog/kl-approx.html for details on this estimator.

    Args:
        q: torch.distribution of shape [batch_size, ...]
        p: torch.distribution of shape [batch_size, ...]

    Returns:
        torch.Tensor: estimated KL divergence.
    """

    # TODO: if q is KumaraswamyStable, use log sample and log prob for stable training

    x = q.rsample((num_samples,))
    logr = p.log_prob(x) - q.log_prob(x)
    kl_ = (logr.exp() - 1) - logr
    # the exponent can cause overflow if the log_prob in p >> log prob in q
    # so remove any rows with inf
    """
    rows_with_inf = torch.isinf(kl_).any(dim=-1)
    if rows_with_inf.any():
        print(f"*** Warning: removing {rows_with_inf.sum()} rows with inf in KL estimation: rows {rows_with_inf.nonzero()}  ***")
    kl_ = kl_[~rows_with_inf]
    """
    kl = kl_.mean(dim=0) # average over samples
    return kl


def scaled_sigmoid(x, lower, upper):
    sigmoid_x = torch.sigmoid(x)
    scaled_x = lower + (upper - lower) * sigmoid_x
    return scaled_x