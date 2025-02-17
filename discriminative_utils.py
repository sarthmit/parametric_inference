import numpy as np
import torch
import torch.nn as nn
from losses import conditional_log_likelihood
from tqdm import tqdm
# from langevin_sampling.samplers import LangevinDynamics
from torch.optim.optimizer import Optimizer

class Optimization:
    def __init__(self, param_size, device):
        self.param_size = param_size
        self.device = device

    def optimize(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, params_mask: torch.Tensor, extra_params, likelihood, conditional_fn, lr: float, iters: int, tqdm: bool = False):
        batch_size = x.shape[1]
        params = nn.Parameter(torch.zeros(batch_size, self.param_size, device=self.device), requires_grad=True)
        nn.init.xavier_uniform_(params)
        optimizer = torch.optim.Adam([params], lr=lr)

        for it in range(iters):
            optimizer.zero_grad()
            loss = conditional_log_likelihood(params, x, y, mask, params_mask, extra_params, likelihood, conditional_fn).sum()
            loss.backward()
            optimizer.step()
        
        return params, loss.item()

    def hparam(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, params_mask: torch.Tensor, extra_params, likelihood, conditional_fn):
        min_lr = None
        min_val = float('inf')
        lrs = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5]

        for lr in lrs:
            _, loss = self.optimize(x, y, mask, params_mask, extra_params, likelihood, conditional_fn, lr, 1000)
            if loss < min_val:
                min_val = loss
                min_lr = lr
            
        return min_lr

    def train(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, params_mask: torch.Tensor, extra_params, likelihood, conditional_fn, num_samples: int = 100, show_tqdm: bool = True, iters: int = 10001):
        lr = self.hparam(x, y, mask, params_mask, extra_params, likelihood, conditional_fn)

        batch_size = x.shape[1]
        params = nn.Parameter(torch.zeros(num_samples, batch_size, self.param_size, device=self.device), requires_grad=True)
        nn.init.xavier_uniform_(params)
        optimizer = torch.optim.Adam([params], lr=lr)
        iterator = tqdm(range(iters), unit="#Samples", ncols=100, leave=True) if show_tqdm else torch.arange(iters)

        for it in iterator:
            optimizer.zero_grad()
            loss = torch.vmap(conditional_log_likelihood, in_dims=0)(
                params,
                x=x,
                y=y,
                mask=mask,
                params_mask=params_mask,
                extra_params=extra_params,
                likelihood=likelihood,
                conditional_fn=conditional_fn
            ).sum()
            loss.backward()
            optimizer.step()
        
        return params

def log_likelihood(params: torch.Tensor, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, params_mask: torch.Tensor, extra_params, likelihood, conditional_fn):
    return conditional_log_likelihood(params, x, y, mask, params_mask, extra_params, likelihood, conditional_fn).sum()

def score_fn(params: torch.Tensor, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, params_mask: torch.Tensor, extra_params, likelihood, conditional_fn):
    grad = -torch.vmap(torch.func.grad(log_likelihood), in_dims=0)(
        params,
        x=x,
        y=y,
        mask=mask,
        params_mask=params_mask,
        extra_params=extra_params,
        likelihood=likelihood,
        conditional_fn=conditional_fn
    )
    grad -= params
    return grad

class Langevin:
    def __init__(self, param_size, device):
        self.param_size = param_size
        self.device = device

    def get_lr(self, it, total):
        return 1e-4 + (1e-6 - 1e-4) * (it / total)

    @torch.no_grad()
    def sample(self, params: torch.Tensor, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, params_mask: torch.Tensor, lr: float, extra_params, likelihood, conditional_fn):
        score = score_fn(params, x, y, mask, params_mask, extra_params, likelihood, conditional_fn)
        noise = torch.randn_like(params)
        params = params + lr * score + np.sqrt(2 * lr) * noise
        return params

    @torch.no_grad()
    def mh_step(self, params):
        return params

    @torch.no_grad()
    def chain(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, params_mask: torch.Tensor, extra_params, likelihood, conditional_fn, num_samples: int = 100, iters: int = 10001, samples_per_chain: int = None, show_tqdm: bool = True):
        batch_size = x.shape[1]
        params = nn.Parameter(torch.zeros(num_samples, batch_size, self.param_size, device=self.device), requires_grad=True)
        nn.init.xavier_uniform_(params)
        iterator = tqdm(range(iters), unit="#Samples", ncols=100, leave=True) if show_tqdm else torch.arange(num_samples)

        parameters = []

        for it in iterator:
            lr = self.get_lr(it, iters)
            params = self.sample(params.detach(), x, y, mask, params_mask, lr, extra_params, likelihood, conditional_fn)
            params = self.mh_step(params)
            if it > iters // 2 and it % 10 == 0:
                parameters.append(params.clone())
                if samples_per_chain is not None:
                    parameters = parameters[-samples_per_chain:]

        parameters = torch.stack(parameters)
        parameters_last = params.clone()
        parameters_single = parameters[:, 0]
        parameters = parameters.view(-1, batch_size, self.param_size)

        return parameters, parameters_last, parameters_single

class HMC:
    def __init__(self, param_size, device):
        self.param_size = param_size
        self.device = device

    def get_lr(self, it, total):
        return 1e-3 + (1e-5 - 1e-3) * (it / total)

    @torch.no_grad()
    def sample(self, params: torch.Tensor, velocity: torch.Tensor, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, params_mask: torch.Tensor, lr: float, extra_params, likelihood, conditional_fn, N: int = 5):
        dlogp = torch.zeros(params.shape[:-1]).to(x.device)
        score = score_fn(params, x, y, mask, params_mask, extra_params, likelihood, conditional_fn)
 
        for _ in range(N):
            # mid point rule
            velocity_half = velocity + 0.5 * score * lr
            params_next = params + velocity_half * lr
          
            # Update change in log probability
            dlogp = dlogp + (score * (params_next - params)).sum(dim=-1)

            # Update momentum and state
            score = score_fn(params_next, x, y, mask, params_mask, extra_params, likelihood, conditional_fn)
            velocity = velocity_half + 0.5 * score * lr
            params = params_next

        return params, velocity, dlogp

    def mh_step(self, params, params_next, velocity, velocity_next, dlogp, M: float = 1.):
        logpvt = -0.5 * (velocity ** 2 / M).sum(dim=-1) # p(vt | xt)
        logpvt_n = -0.5 * (velocity_next ** 2 / M).sum(dim=-1) # p(vt' | xt')
        # Acceptance ratio
        log_alpha = dlogp + logpvt_n - logpvt
        # MH step
        r = torch.rand_like(log_alpha).log()
        accept = r <= log_alpha
        # Update state
        params_next[~accept] = params[~accept]
        return params_next

    @torch.no_grad()
    def chain(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, params_mask: torch.Tensor, extra_params, likelihood, conditional_fn, num_samples: int = 100, iters: int = 10001, samples_per_chain: int = None, show_tqdm: bool = True, M: float = 1.):
        batch_size = x.shape[1]
        params = nn.Parameter(torch.zeros(num_samples, batch_size, self.param_size, device=self.device), requires_grad=True)
        nn.init.xavier_uniform_(params)
        iterator = tqdm(range(iters), unit="#Samples", ncols=100, leave=True) if show_tqdm else torch.arange(num_samples)

        parameters = []

        for it in iterator:
            lr = self.get_lr(it, iters)
            velocity = torch.randn_like(params) * M ** (1/2)
            params_next, velocity_next, dlogp = self.sample(params.detach(), velocity, x, y, mask, params_mask, lr, extra_params, likelihood, conditional_fn)
            params = self.mh_step(params, params_next, velocity, velocity_next, dlogp, M)
            if it > iters // 2 and it % 10 == 0:
                parameters.append(params.clone())
                if samples_per_chain is not None:
                    parameters = parameters[-samples_per_chain:]
        
        parameters = torch.stack(parameters)
        parameters_last = params.clone()
        parameters_single = parameters[:, 0]
        parameters = parameters.view(-1, batch_size, self.param_size)

        return parameters, parameters_last, parameters_single
