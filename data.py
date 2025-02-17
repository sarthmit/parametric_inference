import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gpytorch

class Dataset(nn.Module):
    '''
        Generic Class for generating datasets
    '''
    def __init__(self, dim: int, min_len: int, max_len: int, min_param_len: int, max_param_len: int, mode: str = 'fixed', device=torch.device('cpu')):
        super(Dataset, self).__init__()

        self.mode = mode
        self.min_len = min_len
        self.max_len = max_len
        self.dim = dim

        if self.mode == 'fixed':
            self.min_param_len = self.dim
            self.max_param_len = self.dim
        else:
            self.min_param_len = min_param_len
            self.max_param_len = max_param_len

        self.device = device

    @torch.no_grad()
    def get_mask(self, batch_size: int):
        '''
            Returns: Mask with 1s implying padded values
        '''
        mask = torch.zeros(batch_size, self.max_len).to(self.device)

        lengths = self.min_len + np.random.choice(self.max_len - self.min_len, size=batch_size, replace=True)

        for i, length in enumerate(lengths):
            mask[i, length:] = 1

        return mask

    @torch.no_grad()
    def get_params_mask(self, batch_size: int, length: int = None):
        '''
            Samples Shape: (Sequence_length, Batch_size, dim)
            dims: int
        '''
        mask = torch.zeros(batch_size, self.dim).to(self.device)

        if length is not None:
            mask[:, length:] = 1

        else:
            if self.min_param_len == self.max_param_len:
                lengths = torch.zeros(batch_size, dtype=torch.long) + self.min_param_len
            else:
                lengths = self.min_param_len + \
                    np.random.choice(self.max_param_len - self.min_param_len, size=batch_size, replace=True)

            for i, length in enumerate(lengths):
                mask[i, length:] = 1

        return mask

    def conditional_fn(self, x: torch.Tensor, params: torch.Tensor, params_mask: torch.Tensor):
        '''
            x: (Sequence_length, Batch_size, dim)
            params: (Batch_size, dim)
            params_mask: (Batch_size, dim)
        '''

        return params * (1 - params_mask)
  
    @torch.no_grad()
    def check_fn(self, data: tuple):
        '''
            To ensure that the data generated is not nan
        '''
        if isinstance(data, tuple):
            return torch.isnan(data[0]).any() or torch.isnan(data[1]).any()

        return torch.isnan(data).any()

    @torch.no_grad()
    def sample_batch(self, batch_size: int, length: int = None):
        '''
            Wrapper to ensure no nan data is provided
        '''
        data, data_, params, mask, param_mask = self.get_batch(batch_size, length)

        while self.check_fn(data):
            data, data_, params, mask, param_mask = self.get_batch(batch_size, length)

        return (data, data_, params, mask, param_mask)

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, likelihood, batch_size: int):
        super(ExactGPModel, self).__init__(None, None, likelihood)
        '''
            Module for GP Based data sampling using gpytorch
        '''

        self.mean_module = gpytorch.means.ZeroMean(batch_shape=torch.Size([batch_size]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(lengthscale=1.0, batch_shape=torch.Size([batch_size])),
            batch_shape=torch.Size([batch_size])
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class Gaussian(Dataset):
    '''
        Generating Gaussian Data from randomly sampled means and variances
    '''
    def __init__(self, dim: int, min_len: int, max_len: int, min_param_len: int, max_param_len: int, variance: str = 'identity', mode: str = 'fixed', device=torch.device('cpu')):
        super(Gaussian, self).__init__(dim, min_len, max_len, min_param_len, max_param_len, mode, device)
        self.variance = variance
        self.mean_prob = torch.distributions.normal.Normal(0., 1.)

        if self.variance == 'diagonal':
            self.var_prob = torch.distributions.gamma.Gamma(2., 1.)
        elif self.variance == 'full':
            self.var_prob = torch.distributions.wishart.Wishart(scale_tril=torch.eye(dim), df=dim+1)

    @torch.no_grad()
    def sample_variance(self, batch_size: int):
        '''
            Function to sample variance of a Gaussian Distribution
            Option to sample diagonal or full covariance matrix
        '''
        if self.variance == 'identity':
            var = torch.eye(self.dim).unsqueeze(0).repeat(batch_size, 1, 1).to(self.device)
            param = torch.zeros((batch_size, self.dim)).to(self.device)
        elif self.variance == 'diagonal':
            var = 1. / self.var_prob.rsample((batch_size, self.dim)).to(self.device)
            param = torch.log(var)
            var = torch.diag_embed(var)
        elif self.variance == 'full':
            var = self.var_prob.rsample((batch_size,), max_try_correction=1000).to(self.device)
            var += torch.eye(self.dim).unsqueeze(0).repeat(batch_size, 1, 1).to(self.device)
            var = torch.pinverse(var)
            param = var

        return param, var

    @torch.no_grad()
    def generate_samples(self, params: tuple, params_mask: torch.Tensor):
        '''
            Function to sample a batch of data
        '''
        mean, std = params
        batch_size = mean.shape[0]

        z = torch.randn(self.max_len, batch_size, self.dim).to(self.device)
        samples = mean.unsqueeze(0) + torch.matmul(z.transpose(1, 0), std.transpose(-2, -1)).transpose(1, 0)
        samples = samples * (1-params_mask).unsqueeze(0)
        return samples

    @torch.no_grad()
    def get_batch(self, batch_size: int, length: int = None):
        '''
            Function to generate data
        '''
        mean = self.mean_prob.rsample((batch_size, self.dim)).to(self.device)
        param, var = self.sample_variance(batch_size)
        std, _ = torch.linalg.cholesky_ex(var)

        mask = self.get_mask(batch_size)
        params_mask = self.get_params_mask(batch_size, length)

        train_samples = self.generate_samples((mean, std), params_mask)
        val_samples = self.generate_samples((mean, std), params_mask)

        return train_samples, val_samples, (mean, param), mask.byte(), params_mask.byte()

class LinearRegression(Dataset):
    '''
        Generating Linear Regression Data
    '''
    def __init__(self, dim: int, min_len: int, max_len: int, min_param_len: int, max_param_len: int, x_sampling: str = 'uniform', mode: str = 'fixed', device=torch.device('cpu'), beta: float = 1.):
        super(LinearRegression, self).__init__(dim, min_len, max_len, min_param_len, max_param_len, mode, device)
        self.beta = beta
        self.std = np.sqrt(beta)

        if x_sampling == 'uniform':
            self.x_dist = torch.distributions.uniform.Uniform(-1., 1.)
        else:
            self.x_dist = torch.distributions.normal.Normal(0., 1.)

        self.w_dist = torch.distributions.normal.Normal(0., 1.)

    def conditional_fn(self, x: torch.Tensor, params: torch.Tensor, params_mask: torch.Tensor):
        '''
            x: (Sequence_length, Batch_size, dim)
            params: (Batch_size, dim)
            params_mask: (Batch_size, dim)
        '''
        x = x * (1 - params_mask).unsqueeze(0)
        x = torch.cat([x, torch.ones_like(x[:, :, :1])], dim=-1)
        x = (x * params.unsqueeze(0)).sum(dim=-1, keepdim=True)
        return x

    @torch.no_grad()
    def generate_samples(self, params: torch.Tensor, params_mask: torch.Tensor):
        '''
            Function to sample a batch of data
        '''
        batch_size = params.shape[0]
        x = self.x_dist.rsample((self.max_len, batch_size, self.dim)).to(self.device)
        x = x * (1-params_mask).unsqueeze(0)
        y = self.conditional_fn(x, params, params_mask) + self.std * torch.randn_like(x[:,:,:1])
        return x, y

    @torch.no_grad()
    def get_batch(self, batch_size: int, length: int = None):
        '''
            Function to generate data
        '''
        w = self.w_dist.rsample((batch_size, self.dim + 1)).to(self.device)
        params_mask = self.get_params_mask(batch_size, length)
        mask = self.get_mask(batch_size)

        x_train, y_train = self.generate_samples(w, params_mask)
        x_val, y_val = self.generate_samples(w, params_mask)

        return (x_train, y_train), (x_val, y_val), (w, np.log(self.beta) * torch.ones_like(y_train)), mask.byte(), params_mask.byte()

class LinearClassification(Dataset):
    '''
        Generating Linear Classification Data
    '''
    def __init__(self, dim: int, min_len: int, max_len: int, min_param_len: int, max_param_len: int, num_classes: int = 2, temperature: float = 1., x_sampling: str = 'uniform', mode: str = 'fixed', device=torch.device('cpu')):
        super(LinearClassification, self).__init__(dim, min_len, max_len, min_param_len, max_param_len, mode, device)
        self.num_classes = num_classes
        self.temperature = temperature

        if x_sampling == 'uniform':
            self.x_dist = torch.distributions.uniform.Uniform(-1., 1.)
        else:
            self.x_dist = torch.distributions.normal.Normal(0., 1.)

        self.w_dist = torch.distributions.normal.Normal(0., 1.)

    def conditional_fn(self, x: torch.Tensor, params: torch.Tensor, params_mask: torch.Tensor):
        '''
            x: (Sequence_length, Batch_size, dim)
            params: (Batch_size, dim * num_classes)
            params_mask: (Batch_size, dim)
        '''
        x = x * (1 - params_mask).unsqueeze(0)
        x = torch.cat([x, torch.ones_like(x[:, :, :1])], dim=-1)
        x = x.transpose(1,0)
        params = params.reshape(x.shape[0], self.dim + 1, self.num_classes)
        y = torch.matmul(x, params).transpose(1,0)
        return y

    @torch.no_grad()
    def generate_samples(self, params: torch.Tensor, params_mask: torch.Tensor):
        '''
            Function to sample a batch of data
        '''
        batch_size = params.shape[0]
        x = self.x_dist.rsample((self.max_len, batch_size, self.dim)).to(self.device)
        x = x * (1-params_mask).unsqueeze(0)
        y = self.conditional_fn(x, params, params_mask)
        y = torch.distributions.categorical.Categorical(logits=y / self.temperature).sample().unsqueeze(-1)
        return x, y

    @torch.no_grad()
    def get_batch(self, batch_size: int, length: int = None):
        '''
            Function to generate data
        '''
        w = self.w_dist.rsample((batch_size, (self.dim + 1) * self.num_classes)).to(self.device)
        params_mask = self.get_params_mask(batch_size, length)
        mask = self.get_mask(batch_size)

        x_train, y_train = self.generate_samples(w, params_mask)
        x_val, y_val = self.generate_samples(w, params_mask)

        return (x_train, y_train), (x_val, y_val), (w, None), mask.byte(), params_mask.byte()

class NonLinearRegression(Dataset):
    '''
        Generating Non-Linear Regression Data
    '''
    def __init__(self, dim: int, min_len: int, max_len: int, min_param_len: int, max_param_len: int, h_dim: int, n_layers: int, act, x_sampling: str = 'uniform', mode: str = 'fixed', device=torch.device('cpu'), beta: float = 1.):
        super(NonLinearRegression, self).__init__(dim, min_len, max_len, min_param_len, max_param_len, mode, device)
        self.h_dim = h_dim
        self.param_size = (dim + 1) * h_dim + h_dim + (h_dim * h_dim * (n_layers - 1))
        self.n_layers = n_layers
        self.act = act
        self.std = np.sqrt(beta)
        self.beta = beta

        if x_sampling == 'uniform':
            self.x_dist = torch.distributions.uniform.Uniform(-1., 1.)
        else:
            self.x_dist = torch.distributions.normal.Normal(0., 1.)

        self.w_dist = torch.distributions.normal.Normal(0., 1.)

    def conditional_fn(self, x: torch.Tensor, params: torch.Tensor, params_mask: torch.Tensor):
        '''
            x: (Sequence_length, Batch_size, dim)
            params: (Batch_size, dims)
            params_mask: (Batch_size, dim)
        '''
        x = x * (1 - params_mask).unsqueeze(0)
        h = torch.cat([x, torch.ones_like(x[:, :, :1])], dim=-1)
        h = h.transpose(1, 0).contiguous()
        batch_size, _, dim = h.shape

        h = self.act(torch.matmul(h, params[:, :dim * self.h_dim].reshape(batch_size, dim, self.h_dim)))
        start = dim * self.h_dim
        for _ in range(self.n_layers - 1):
            h = self.act(torch.matmul(h, params[:, start: start + self.h_dim * self.h_dim].reshape(batch_size, self.h_dim, self.h_dim)))
            start += self.h_dim * self.h_dim

        h = torch.matmul(h, params[:, start: start + self.h_dim].reshape(batch_size, self.h_dim, 1))
        h = h.transpose(1, 0).contiguous()
        return h

    @torch.no_grad()
    def generate_samples(self, params: torch.Tensor, params_mask: torch.Tensor):
        '''
            Function to sample a batch of data
        '''
        batch_size = params.shape[0]
        x = self.x_dist.rsample((self.max_len, batch_size, self.dim)).to(self.device)
        x = x * (1-params_mask).unsqueeze(0)
        y = self.conditional_fn(x, params, params_mask) + self.std * torch.randn_like(x[:,:,:1])
        return x, y

    @torch.no_grad()
    def get_batch(self, batch_size: int, length: int = None):
        '''
            Function to generate data
        '''
        w = self.w_dist.rsample((batch_size, self.param_size)).to(self.device)
        params_mask = self.get_params_mask(batch_size, length)
        mask = self.get_mask(batch_size)

        x_train, y_train = self.generate_samples(w, params_mask)
        x_val, y_val = self.generate_samples(w, params_mask)

        return (x_train, y_train), (x_val, y_val), (w, np.log(self.beta) * torch.ones_like(y_train)), mask.byte(), params_mask.byte()

class GP(Dataset):
    '''
        Generating Non-Linear GP Data
    '''

    def __init__(self, dim: int, min_len: int, max_len: int, min_param_len: int, max_param_len: int, x_sampling: str = 'uniform', mode: str = 'fixed', device=torch.device('cpu')):
        super(GP, self).__init__(dim, min_len, max_len, min_param_len, max_param_len, mode, device)
        self.min_len = min_len
        self.max_len = max_len
        self.dim = dim
        self.device = device

        batch_size = 128
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-8), batch_shape=torch.Size([batch_size])).eval()
        self.likelihood.initialize(noise=1e-6)
        self.model = ExactGPModel(self.likelihood, batch_size).eval().to(device)

        self.x_dist = torch.distributions.normal.Normal(0., 1.)

        if x_sampling == 'uniform':
            self.x_dist = torch.distributions.uniform.Uniform(-1., 1.)
        else:
            self.x_dist = torch.distributions.normal.Normal(0., 1.)

    @torch.no_grad()
    def get_batch(self, batch_size: int, length: int = None):
        '''
            Function to generate data
        '''
        x = self.x_dist.rsample((128, 2 * self.max_len, self.dim)).to(self.device)
        y = self.likelihood(self.model(x)).rsample().unsqueeze(-1)

        x_train, x_val = x[:batch_size, :self.max_len, :].transpose(1,0), x[:batch_size, self.max_len:, :].transpose(1,0)
        y_train, y_val = y[:batch_size, :self.max_len, :].transpose(1,0), y[:batch_size, self.max_len:, :].transpose(1,0)

        mask = self.get_mask(batch_size)
        params_mask = self.get_params_mask(batch_size, length)

        return (x_train, y_train), (x_val, y_val), (None, None), mask.byte(), params_mask.byte()

class NonLinearClassification(Dataset):
    '''
        Generating Non-Linear Classification Data
    '''
    def __init__(self, dim: int, min_len: int, max_len: int, min_param_len: int, max_param_len: int, h_dim: int, n_layers: int, act, num_classes: int = 2, temperature: float = 1., x_sampling: str = 'uniform', mode: str = 'fixed', device=torch.device('cpu')):
        super(NonLinearClassification, self).__init__(dim, min_len, max_len, min_param_len, max_param_len, mode, device)
        self.h_dim = h_dim
        self.param_size = (dim+1) * h_dim + h_dim * num_classes + (h_dim * h_dim * (n_layers - 1))
        self.n_layers = n_layers
        self.act = act
        self.temperature = temperature
        self.num_classes = num_classes

        if x_sampling == 'uniform':
            self.x_dist = torch.distributions.uniform.Uniform(-1., 1.)
        else:
            self.x_dist = torch.distributions.normal.Normal(0., 1.)

        self.w_dist = torch.distributions.normal.Normal(0., 1.)

    def conditional_fn(self, x: torch.Tensor, params: torch.Tensor, params_mask: torch.Tensor):
        '''
            x: (Sequence_length, Batch_size, dim)
            params: (Batch_size, dims)
            params_mask: (Batch_size, dim)
        '''
        x = x * (1 - params_mask).unsqueeze(0)
        h = torch.cat([x, torch.ones_like(x[:, :, :1])], dim=-1)
        h = h.transpose(1, 0).contiguous()
        batch_size, _, dim = h.shape

        h = self.act(torch.matmul(h, params[:, :dim * self.h_dim].reshape(batch_size, dim, self.h_dim)))
        start = dim * self.h_dim
        for _ in range(self.n_layers - 1):
            h = self.act(torch.matmul(h, params[:, start: start + self.h_dim * self.h_dim].reshape(batch_size, self.h_dim, self.h_dim)))
            start += self.h_dim * self.h_dim

        h = torch.matmul(h, params[:, start: start + self.h_dim * self.num_classes].reshape(batch_size, self.h_dim, self.num_classes))
        h = h.transpose(1, 0).contiguous()
        return h

    @torch.no_grad()
    def generate_samples(self, params: torch.Tensor, params_mask: torch.Tensor):
        '''
            Function to sample a batch of data
        '''
        batch_size = params.shape[0]
        x = self.x_dist.rsample((self.max_len, batch_size, self.dim)).to(self.device)
        x = x * (1-params_mask).unsqueeze(0)
        y = self.conditional_fn(x, params, params_mask)
        y = torch.distributions.categorical.Categorical(logits=y / self.temperature).sample().unsqueeze(-1)
        return x, y

    @torch.no_grad()
    def get_batch(self, batch_size: int, length: int = None):
        '''
            Function to generate data
        '''
        w = self.w_dist.rsample((batch_size, self.param_size)).to(self.device)
        params_mask = self.get_params_mask(batch_size, length)
        mask = self.get_mask(batch_size)

        x_train, y_train = self.generate_samples(w, params_mask)
        x_val, y_val = self.generate_samples(w, params_mask)

        return (x_train, y_train), (x_val, y_val), (w, None), mask.byte(), params_mask.byte()

class GMM(Dataset):
    '''
        Generating GMM Data
    '''
    def __init__(self, dim: int, n_mixtures: int, min_len: int, max_len: int, min_param_len: int, max_param_len: int, variance: str = 'identity', sample_mixture_weights: bool = False, mode: str = 'fixed', device=torch.device('cpu')):
        super(GMM, self).__init__(dim, min_len, max_len, min_param_len, max_param_len, mode, device)
        self.n_mixtures = n_mixtures
        self.variance = variance
        self.sample_mixture_weights = sample_mixture_weights

        self.mean_prob = torch.distributions.normal.Normal(0., 1.)

        if self.variance == 'diagonal':
            self.var_prob = torch.distributions.gamma.Gamma(2., 1.)
        elif self.variance == 'full':
            self.var_prob = torch.distributions.wishart.Wishart(scale_tril=torch.eye(dim), df=dim+1)

        if self.sample_mixture_weights:
            self.pi_prob = torch.distributions.dirichlet.Dirichelt(concentration=torch.ones(n_mixtures))

    def sample_variance(self, batch_size: int):
        '''
            Function to sample variance of a Gaussian Distribution
            Option to sample diagonal or full covariance matrix
        '''
        if self.variance == 'identity':
            var = 0.1 * torch.eye(self.dim).unsqueeze(0).unsqueeze(0).repeat(batch_size, self.n_mixtures, 1, 1).to(self.device)
            param = torch.zeros((batch_size, self.n_mixtures, self.dim)).to(self.device) + np.log(0.1)
        elif self.variance == 'diagonal':
            var = 0.1 / self.var_prob.rsample((batch_size, self.n_mixtures, self.dim)).to(self.device)
            param = torch.log(var)
            var = torch.diag_embed(var)
        elif self.variance == 'full':
            var = self.var_prob.rsample((batch_size * self.n_mixtures,), max_try_correction=1000).to(self.device).view(batch_size, self.n_mixtures, self.dim, self.dim)
            var += torch.eye(self.dim).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1).to(self.device)
            var = 0.1 * torch.pinverse(var)
            param = var

        return param, var

    def conditional_fn(self, x: torch.Tensor, params: torch.Tensor, params_mask: torch.Tensor):
        '''
            x: (Sequence_length, Batch_size, dim)
            params: (Batch_size, dim)
            params_mask: (Batch_size, dim)
        '''
        batch_size = params.shape[0]
        params = params.view(batch_size, self.n_mixtures, self.dim) * (1 - params_mask).unsqueeze(1)
        params = params.view(batch_size, -1)
        return params

    @torch.no_grad()
    def generate_samples(self, params: tuple, params_mask: torch.Tensor):
        '''
            Function to sample a batch of data
        '''
        pi, mean, std = params
        batch_size = mean.shape[1]

        mode = torch.multinomial(pi, self.max_len, replacement=True).view(batch_size, self.max_len).to(self.device)
        z = torch.randn(self.max_len, 1, batch_size, self.dim).to(self.device).repeat(1, self.n_mixtures, 1, 1)
        one_hot_mode = F.one_hot(mode, num_classes=self.n_mixtures)
        samples = mean.unsqueeze(0) + torch.matmul(z.transpose(2, 0), std.transpose(-2, -1)).transpose(2, 0)
        samples = (samples * one_hot_mode.permute(1, 2, 0).unsqueeze(-1)).sum(dim=1)
        samples = samples * (1-params_mask).unsqueeze(0)
        return samples

    @torch.no_grad()
    def get_batch(self, batch_size: int, length: int = None):
        '''
            Function to generate data
        '''
        if self.sample_mixture_weights:
            pi = self.pi_prob.rsample((batch_size,)).to(self.device)
        else:
            pi = torch.ones((batch_size, self.n_mixtures)).to(self.device) / self.n_mixtures
        
        mean = self.mean_prob.rsample((self.n_mixtures, batch_size, self.dim)).to(self.device)
        param, var = self.sample_variance(batch_size)
        std, _ = torch.linalg.cholesky_ex(var)

        mask = self.get_mask(batch_size)
        params_mask = self.get_params_mask(batch_size, length)

        train_samples = self.generate_samples((pi, mean, std), params_mask)
        val_samples = self.generate_samples((pi, mean, std), params_mask)

        return train_samples, val_samples, (mean.transpose(1, 0).reshape(batch_size, -1), (param, pi)), mask.byte(), params_mask.byte()