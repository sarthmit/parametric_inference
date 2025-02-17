import torch
import numpy as np
from distributions import GaussianDistribution
from utils import check_psd

def posterior_mean_full_gaussian(samples, mask, variance, device=torch.device('cuda')):
    _, batch_size, dim = samples.shape
    if len(variance.shape) <=2:
        variance = torch.diag_embed(torch.exp(variance))
    var_inv = torch.pinverse(variance)

    samples = (samples * (1-mask).transpose(1,0).unsqueeze(-1))
    count = (1-mask).sum(dim=1)

    posterior_variance = count[:, None, None] * var_inv + torch.eye(dim).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
    posterior_variance = torch.pinverse(posterior_variance)
    posterior_mean = torch.matmul(samples.sum(dim=0).unsqueeze(1), var_inv)
    posterior_mean = torch.matmul(posterior_mean, posterior_variance).squeeze(1)

    return GaussianDistribution((posterior_mean, posterior_variance))

def posterior_linear_regression(x, y, mask, beta, device=torch.device('cuda')):
    x = torch.cat([x, torch.ones_like(x[:,:,:1])], dim=-1)
    x = x.transpose(1, 0)
    y = y.transpose(1, 0)
    batch_size, _, dim = x.shape

    x_s = x * (1-mask).unsqueeze(-1)
    y_s = y * (1-mask).unsqueeze(-1)
    x_st = x_s.transpose(-2, -1)
    posterior_variance = torch.matmul(x_st, x_s) / beta + torch.eye(dim).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
    posterior_variance = torch.pinverse(posterior_variance)
    posterior_mean = torch.matmul(posterior_variance, torch.matmul(x_st, y_s) / beta).squeeze(-1)

    return GaussianDistribution((posterior_mean, posterior_variance))

def test_posterior_mean_full_gaussian():
    x = torch.randn([3, 5000, 2])

    var = 1. / torch.distributions.gamma.Gamma(1., 1.).rsample((3, 2))
    std = torch.sqrt(var)
    mean = torch.randn(3, 2)

    x = mean[:, None, :] + std[:, None, :] * x

    posterior_mean, posterior_var = posterior_mean_normal(x, torch.zeros_like(x[:, :, 0]), torch.diag_embed(var))
    print(posterior_mean)
    print(mean)

def test_posterior_linear_regression():
    x = torch.randn(32, 3, 1)
    w = torch.randn(3, 2)
    y = (torch.cat([x, torch.ones_like(x[:,:,:1])], dim=-1) * w.unsqueeze(0)).sum(dim=-1, keepdim=True) + 0.1 * torch.randn_like(x[:,:,:1])
    x_s = torch.linspace(x.min(), x.max(), steps=2)
    masks = torch.zeros(3, 32).byte()

    posterior = posterior_linear_regression(x, y, masks, 0.1, device=torch.device('cpu'))
    idx = np.random.choice(3)

    import matplotlib.pyplot as plt
    plt.scatter(x[:,idx,0], y[:, idx, 0], s=5, alpha=0.25)
    predict = (torch.cat([x, torch.ones_like(x[:,:,:1])], dim=-1) * posterior.mean.unsqueeze(0)).sum(dim=-1, keepdim=True)
    plt.plot(x[:, idx, 0], predict[:, idx, 0])
    plt.savefig('test.png')

if __name__ == '__main__':
    test_posterior_linear_regression()
