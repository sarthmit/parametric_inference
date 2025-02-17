import torch
from losses import conditional_log_likelihood, kl_logvar
from distributions import CategoricalDistribution
from utils import one_hot
import wandb
from score_wrapper import *

def preprocess_samples(samples, num_classes: int = None):
    if isinstance(samples, tuple):
        '''
            If supervised learning problem, get inputs and targets
        '''
        x, y = samples

        if isinstance(y, torch.cuda.LongTensor):
            '''
                If classification, perform one-hot encoding
            '''
            samples = torch.cat((x, one_hot(y, num_classes)), dim=-1)
        else:
            samples = torch.cat(samples, dim=-1)

    else:
        '''
            If unsupervised learning problem, set inputs and targets to be the same
        '''
        x, y = samples, samples
    
    return x, y, samples

def reverse_kl(approx_posterior, x, y, mask, params_mask, likelihood, extra_params, prior, it, conditional_fn, model_choice: str = 'Vanilla', warmup_iters: int = 1):
    '''
        Train to minimize the reverse KL Divergence between true and approximate posterior
    '''
    if model_choice == 'Flow':
        sample, z, ldj = approx_posterior.sample(only_sample=False)
        loss = min(1., (it + 1) / warmup_iters) * (prior.nll(sample) - ldj - prior.nll(z))
    else:
        sample = approx_posterior.sample()
        loss = min(1., (it + 1) / warmup_iters) * kl_logvar(approx_posterior, prior)

    cll = conditional_log_likelihood(approx_posterior.sample(), x, y,
                                    mask, params_mask, extra_params=extra_params,
                                    conditional_fn=conditional_fn,
                                    likelihood=likelihood)

    loss = (cll + loss).mean()
    return loss

def forward_kl(approx_posterior, params, prior, model_choice: str = 'Vanilla'):
    '''
        Train to minimize the foward KL Divergence between true and approximate posterior
    '''
    if model_choice == 'Flow':
        z, ldj = approx_posterior.reverse(params)
        loss = prior.nll(z)
    else:
        ldj = 0
        loss = approx_posterior.nll(params)

    loss = (loss - ldj).mean()
    return loss

def mle(approx_posterior, x, y, mask, params_mask, likelihood, extra_params, conditional_fn):
    '''
        Train to maximize the log likelihood of the data
    '''
    cll = conditional_log_likelihood(approx_posterior, x, y,
                                    mask, params_mask, extra_params=extra_params,
                                    conditional_fn=conditional_fn,
                                    likelihood=likelihood)

    return cll.mean()

def map(approx_posterior, x, y, mask, params_mask, likelihood, extra_params, conditional_fn):
    '''
        Train to get the MAP estimate
    '''
    loss = mle(approx_posterior, x, y, mask, params_mask, likelihood, extra_params, conditional_fn)
    loss += 0.5 * (approx_posterior ** 2).sum(dim=-1).mean(dim=0)
    return loss

def diffusion(approx_posterior, params, noise_schedule):
    '''
        Train to get the score estimate
    '''
    t = torch.rand(params.shape[0]).to(params.device)
    std = noise_schedule.h(t).unsqueeze(-1) ** 0.5
    std_derivative = noise_schedule.f(t).unsqueeze(-1)

    z = torch.randn_like(params)
    noised_params = params + std * z
    score = approx_posterior.score(t, noised_params)
    loss = 0.5 * (score + 2 * std_derivative * z).pow(2).mean()
    return loss

def flow_matching(approx_posterior, params):
    '''
        Train to get the score estimate
    '''
    t = torch.rand(params.shape[0]).to(params.device).unsqueeze(-1)
    z = torch.randn_like(params)
    noised_params = (1. - t) * params + t * z
    cond_drift = z - params
    drift = approx_posterior.drift(t.squeeze(-1), noised_params)
    loss = 0.5 * (drift - cond_drift).pow(2).mean()
    return loss

def idem(approx_posterior, x, y, mask, params_mask, likelihood, extra_params, conditional_fn, noise_schedule, num_mc_samples: int = 100):
    '''
        Train to get the score estimate
    '''
    def estimate(params, t, x, y, mask, params_mask, extra_params):
        noised_params = params.unsqueeze(0) + (noise_schedule.h(t).unsqueeze(0).unsqueeze(-1) ** 0.5) * torch.randn([num_mc_samples, params.shape[0], params.shape[1]]).to(params.device)
        loss = -torch.vmap(conditional_log_likelihood, in_dims=0)(noised_params, x=x, y=y, mask=mask, params_mask=params_mask, extra_params=extra_params, conditional_fn=conditional_fn, likelihood=likelihood)
        loss += -0.5 * (noised_params ** 2).sum(dim=-1)
        return torch.logsumexp(loss, dim=0).sum()
    
    params = (noise_schedule.h(1) ** 0.5) * torch.randn([x.shape[1], approx_posterior.dim]).to(x.device)
    t = torch.rand(params.shape[0]).to(params.device)
    score_estimate = torch.func.grad(estimate)(params, t, x, y, mask, params_mask, extra_params)
    diff_coeff = noise_schedule.g(t).unsqueeze(-1) ** 2

    score_norms = torch.linalg.vector_norm(score_estimate, dim=-1).detach()

    clip_coefficient = torch.clamp(
        50. / (score_norms + 1e-4), max=1
    )

    score_estimate = score_estimate * clip_coefficient.unsqueeze(-1)

    score = approx_posterior.score(t, params)

    loss = 0.5 * (score - diff_coeff * score_estimate).pow(2).mean()
    return loss

def train(batch, model, optimizer, conditional_fn, likelihood, prior_args, it, noise_schedule, objective: str = 'forward', model_choice: str = 'Vanilla', warmup_iters: int = 1, num_classes: int = None, log_wandb: bool = False):
    '''
        Main function that decides which loss function to use
    '''
    samples, _, params, mask, params_mask = batch
    x, y, samples = preprocess_samples(samples, num_classes)
    prior = prior_args['dist']((prior_args['mean'].repeat(samples.shape[1], 1),
                              prior_args['logvar'].repeat(samples.shape[1], 1)))
    approx_posterior = model(samples, mask)

    if objective == 'forward':
        loss = forward_kl(approx_posterior, params[0], prior, model_choice=model_choice)
    elif objective == 'backward':
        loss = reverse_kl(approx_posterior, x, y, mask, params_mask, likelihood, (params[1],), prior, it, conditional_fn, model_choice, warmup_iters)
    elif objective == 'mixed':
        loss_fwd = forward_kl(approx_posterior, params[0], prior, model_choice=model_choice)
        loss_rev = reverse_kl(approx_posterior, x, y, mask, params_mask, likelihood, (params[1],), prior, it, conditional_fn, model_choice, warmup_iters)
        if log_wandb:
            wandb.log({
                "Train | Forward KL": loss_fwd.item(),
                "Train | Reverse KL": loss_rev.item()
            }, step=it)
        loss = 0.5 * (loss_fwd + loss_rev)
    elif objective == 'mle':
        loss = mle(approx_posterior, x, y, mask, params_mask, likelihood, (params[1],), conditional_fn)
    elif objective == 'map':
        loss = map(approx_posterior, x, y, mask, params_mask, likelihood, (params[1],), conditional_fn)
    elif objective == 'diffusion':
        loss = diffusion(approx_posterior, params[0], noise_schedule=noise_schedule)
    elif objective == 'flow-matching':
        loss = flow_matching(approx_posterior, params[0])
    elif objective == 'idem':
        loss = idem(approx_posterior, x, y, mask, params_mask, likelihood, (params[1],), conditional_fn, noise_schedule=noise_schedule)
    loss.backward()
    optimizer.step()

    if log_wandb:
        wandb.log({
            "Train | Loss": loss.item(),
        }, step=it)

    return loss.item()

def train_step(model, optimizer, dataset, conditional_fn, likelihood, prior_args, args, noise_schedule, it):
    model.train()
    model.zero_grad()

    batch = dataset.sample_batch(args.batch_size)
    loss = train(batch, model, optimizer, conditional_fn, likelihood, prior_args, it, noise_schedule, args.objective, args.model, args.warmup_iters, num_classes=args.num_classes, log_wandb=args.wandb)
    return loss