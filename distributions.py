import torch
import numpy as np

class GaussianDistribution():
  def __init__(self, params, unsupervised=False):
    self.unsupervised = unsupervised
    self.loc, self.var = params[0], params[1]
    self.std, _ = torch.linalg.cholesky_ex(self.var)
    self.var_inv = torch.linalg.pinv(self.var)

  @property
  def mean(self):
    return self.loc

  @property
  def cov(self):
    return self.var

  @property
  def mode(self):
    return self.mean

  @property
  def dim(self):
    return self.mean.shape[-1]

  def sample(self, size=None):
    if size is None:
      size = []
      size = torch.Size(size)
    
    z = torch.randn(size + self.mean.size()).to(self.mean.device)
    return self.mean + torch.matmul(self.std, z.unsqueeze(-1)).squeeze(-1)

  def nll(self, sample, params_mask=None):
    x_ = torch.matmul(self.var_inv, (sample - self.mean).unsqueeze(-1)).squeeze(-1)
    normalizer = torch.logdet(self.cov) + self.dim * np.log(2 * np.pi)
    nll = -0.5 * normalizer - 0.5 * ((sample - self.mean) * x_).sum(dim=-1)
    return -nll


class DiagonalGaussianDistribution():
  def __init__(self, params, unsupervised=False):
    self.unsupervised = unsupervised
    if isinstance(params, torch.Tensor):
      params = (params, ) + (torch.zeros_like(params) - 2.3, )
    mean, logvar = params
    self.mean = mean
    self.logvar = logvar

  @property
  def cov(self):
    return torch.diag_embed(torch.exp(self.logvar))

  @property
  def mode(self):
    return self.mean

  def sample(self, size=None):
    sample = torch.randn_like(self.mean)
    sample = self.mean + torch.exp(self.logvar / 2.) * sample
    return sample

  def nll(self, sample, params_mask=None):
    nll = -0.5 * (torch.exp(-self.logvar) * (self.mean - sample) ** 2 + self.logvar + np.log(2 * np.pi))
    if self.unsupervised:
      nll = nll * (1-params_mask).unsqueeze(0)
    nll = nll.sum(dim=-1)
    return -nll


class MixtureDiagonalGaussianDistribution():
  def __init__(self, params, unsupervised=False):
    self.unsupervised = unsupervised
    mean, (logvar, weights) = params
    self.mean = mean
    self.logvar = logvar
    self.weights = weights

    self.n_mixtures = weights.shape[-1]
    self.mean = self.mean.view(self.mean.shape[0], self.n_mixtures, self.dim)

  @property
  def dim(self):
    return self.logvar.shape[-1]

  def nll(self, sample, params_mask=None):
    # sample: (max_len, bsz, dim)
    # mean: (bsz, n_mixtures, dim)
    # weights: (bsz, n_mixtures)
    # logvar: (bsz, n_mixtures, dim)

    nll = -0.5 * (torch.exp(-self.logvar) * (self.mean.unsqueeze(0) - sample.unsqueeze(-2)) ** 2 + self.logvar + np.log(2 * np.pi))
    if self.unsupervised:
        nll = nll * (1-params_mask).unsqueeze(0).unsqueeze(-2)
    nll = nll.sum(dim=-1)
    nll += self.weights.unsqueeze(0).log()
    nll = torch.logsumexp(nll, dim=-1)
    return -nll


class MixtureGaussianDistribution():
  def __init__(self, params, unsupervised=False):
    self.unsupervised = unsupervised
    mean, (var, weights) = params
    self.mean = mean
    self.var = var
    self.weights = weights

    self.n_mixtures = weights.shape[-1]
    self.mean = self.mean.view(self.mean.shape[0], self.n_mixtures, self.dim)

    self.std, _ = torch.linalg.cholesky_ex(self.var)
    self.var_inv = torch.linalg.pinv(self.var)

  @property
  def dim(self):
    return self.var.shape[-1]

  def nll(self, sample, params_mask=None):
    # sample: (max_len, bsz, dim)
    # mean: (bsz, n_mixtures, dim)
    # weights: (bsz, n_mixtures)
    # var_inv: (bsz, n_mixtures, dim, dim)

    x_ = torch.matmul(self.var_inv, (sample.unsqueeze(-2) - self.mean.unsqueeze(0)).unsqueeze(-1)).squeeze(-1)
    nll = -0.5 * self.dim * np.log(2 * np.pi) - 0.5 * torch.logdet(self.var) - \
          0.5 * ((sample.unsqueeze(-2) - self.mean.unsqueeze(0)) * x_).sum(dim=-1)
    nll += self.weights.unsqueeze(0).log()
    nll = torch.logsumexp(nll, dim=-1)
    return -nll


class FlowDiagonalGaussianDistribution():
  def __init__(self, start_dist, params, flow):
    self.h1, self.h2 = params
    self.start_dist = start_dist
    self.cinn = flow

  @property
  def mode(self):
    mode = self.start_dist.mode
    mode, _ = self.cinn(mode, [self.h1, self.h2])
    return mode

  def sample(self, size=None, only_sample=True):
    z = self.start_dist.sample()
    samples, ldj = self.cinn(z, [self.h1, self.h2])
    if only_sample:
      return samples

    return samples, z, ldj

  def start_nll(self, sample):
    return self.start_dist.nll(sample)

  def reverse(self, sample):
    samples, ldj = self.cinn(sample, [self.h1, self.h2], rev=True)
    return samples, ldj

class CategoricalDistribution():
  def __init__(self, logits):
    self.logits = logits
    self.dist = torch.distributions.categorical.Categorical(logits=logits, validate_args=False)

  @property
  def mean(self):
    return self.dist.mean

  @property
  def cov(self):
    return self.dist.variance

  @property
  def mode(self):
    return self.dist.mode

  def sample(self):
    return self.dist.sample()

  def nll(self, sample, params_mask=None):
    return -self.dist.log_prob(sample.squeeze(-1))

if __name__ == '__main__':
  mean = torch.randn(32, 3, 5)
  var = torch.eye(5).unsqueeze(0).unsqueeze(0).repeat(32, 3, 1, 1)
  weights = torch.ones(32, 3) / 3.

  dist = MixtureGaussianDistribution((mean.unsqueeze(0), var, weights))
  dist.nll(torch.randn(64, 32, 5))