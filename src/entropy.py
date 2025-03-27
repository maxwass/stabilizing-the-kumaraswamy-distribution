import torch
from torch.special import digamma, polygamma

""" Unused. Have not been able to get this to work better than simply using regular torch.distribution.Kumaraswamy.entropy after exponentiating the log parameters."""
dtype = torch.float32

MAX_GRAD_MAGNITUDE = 1e35
EULER_CONSTANT = torch.tensor(0.57721566490153286060, dtype=dtype) # Euler-Mascheroni Constant

def harmonic_number_func(x: torch.tensor):
    # Confirm input is positive
    assert torch.all(x > 0), "Input must be positive"
    # Compute harmonic number using digamma function
    return torch.digamma(x + 1) + EULER_CONSTANT

def kumaraswamy_entropy_reparam(alpha, beta):
    return (1 - beta) + (1-alpha) * harmonic_number_func(1/beta) + (torch.log(alpha) + torch.log(beta))

def kumaraswamy_entropy(a, b):
    a_inv, b_inv = 1/a, 1/b
    return (1 - b_inv) + (1 - a_inv) * harmonic_number_func(b) - (torch.log(a) + torch.log(b))

def kumaraswamy_entropy_reparam_gradient(alpha, beta):
    # NEED TO TEST, NOT CORRECT?!
    alpha_inv, beta_inv = 1/alpha, 1/beta
    nabla_alpha = harmonic_number_func(beta_inv) + alpha_inv
    nabla_beta = -1 - (1-alpha) * torch.polygamma(1, beta_inv + 1) * beta_inv**2 + beta_inv
    return nabla_alpha, nabla_beta

def kumaraswamy_moment(a, b, n):
    gamma_1_a = torch.lgamma(1 + n / a)  # Gamma(1 + n/a)
    gamma_b = torch.lgamma(b)            # Gamma(b)
    gamma_1_a_b = torch.lgamma(1 + n / a + b)  # Gamma(1 + n/a + b)

    moment = torch.exp(torch.log(b) + gamma_1_a + gamma_b - gamma_1_a_b)
    return moment

def kumaraswamy_mean(a, b):
    return kumaraswamy_moment(a, b, 1)
class KumaraswamyStableEntropyOLD(torch.autograd.Function):
    # more stable than pytorch implemenation.
    # stable in float32 for \log_2 a \in [-45, 100], \log_2 b \in [-100, 100]

    # see kumar_entropy.ipynb for details

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor): # INPUTS = a, b, NOT log_a, log_b
        a_inv, b_inv = 1/a, 1/b

        ctx.save_for_backward(b, a_inv, b_inv)
        return (1 - b_inv) + (1 - a_inv) * (digamma(b + 1) + EULER_CONSTANT) - torch.log(a) - torch.log(b)

    @staticmethod
    def backward(ctx, grad_output):
        b, a_inv, b_inv = ctx.saved_tensors

        nabla_a = a_inv * ( a_inv * (digamma(b + 1.0) + EULER_CONSTANT) - 1.0 )
        nabla_a = nabla_a.clamp(min=-MAX_GRAD_MAGNITUDE, max=MAX_GRAD_MAGNITUDE)

        nabla_b = b_inv**2 + (1 - a_inv) *  (polygamma(1, b + 1.0)) - b_inv
        nabla_b = nabla_b.clamp(min=-MAX_GRAD_MAGNITUDE, max=MAX_GRAD_MAGNITUDE)

        # Compute gradients
        grad_a = grad_b = None

        if ctx.needs_input_grad[0]:
            grad_a = grad_output * nabla_a
        if ctx.needs_input_grad[1]:
            grad_b = grad_output * nabla_b
        
        return grad_a, grad_b 

class KumaraswamyStableEntropy(torch.autograd.Function):

    @staticmethod
    def forward(ctx, log_a: torch.Tensor, log_b: torch.Tensor):
        a_inv, b_inv, b = exp(-log_a), exp(-log_b), exp(log_b)
        digamma_euler = digamma(b + 1) + EULER_CONSTANT

        ctx.save_for_backward(b, a_inv, b_inv, digamma_euler)
        return (1 - b_inv) + (1 - a_inv) * digamma_euler - log_a - log_b

    @staticmethod
    def backward(ctx, grad_output):
        b, a_inv, b_inv, digamma_euler = ctx.saved_tensors

        grad_log_a = grad_log_b = None

        if ctx.needs_input_grad[0]:
            nabla_log_a = a_inv * digamma_euler - 1.0
            grad_log_a = grad_output * nabla_log_a
        if ctx.needs_input_grad[1]:
            # TODO: (1-a_inv) may be unstable...Can test with ~ log1mexp( log_a )
            #stable_1ma_inv = (1-a_inv).sign() * log1mexp(log_a.abs())# unsure if correct
            nabla_log_b = b_inv + b * (1-a_inv) * polygamma(1, b + 1.0) - 1.0
            grad_log_b = grad_output * nabla_log_b
        
        return grad_log_a, grad_log_b 

kumaraswamy_stable_entropy = KumaraswamyStableEntropy.apply
