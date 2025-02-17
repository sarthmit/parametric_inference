import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from architectures import *
from distributions import DiagonalGaussianDistribution, FlowDiagonalGaussianDistribution
from score_wrapper import ScoreWrapper, FlowWrapper
from functools import partial

class GaussianModel(nn.Module):
    '''
        Model for using Gaussian distribution as the posterior approximation
    '''
    def __init__(self, encoder: str, model_args):
        super(GaussianModel, self).__init__()
        self.encoder = eval(encoder)(**model_args[encoder])

    def posterior_diagonal_gaussian(self, params, clamp=True):
        mean, logvar = torch.chunk(params, 2, dim=-1)
        if clamp:
            logvar = torch.clamp(logvar, -30.0, 20.0)
        return DiagonalGaussianDistribution((mean, logvar))
    
    def sample(self, x: torch.Tensor = None, mask: torch.Tensor = None, distribution = None):
        if distribution is None:
            params = self.encoder(x, mask)
            distribution = self.posterior_diagonal_gaussian(params)
        return distribution.sample()

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        params = self.encoder(x, mask)
        return self.posterior_diagonal_gaussian(params)

class FlowModel(GaussianModel):
    '''
        Model for using Normalizing Flow as the posterior approximation
    '''
    def __init__(self, encoder: str, model_args):
        super(FlowModel, self).__init__(encoder, model_args)
        self.flow_model = SequentialNF(**model_args['Posterior_Flow'])

    def posterior_flow_model(self, params: torch.Tensor, flow_model):
        h1, h2 = torch.chunk(params, 2, dim=-1)
        latent_dist = self.posterior_diagonal_gaussian(torch.zeros_like(h1))
        return FlowDiagonalGaussianDistribution(latent_dist, (h1, h2), flow_model)

    def sample(self, x: torch.Tensor = None, mask: torch.Tensor = None, distribution = None, only_sample: bool = False):
        if distribution is None:
            params = self.encoder(x, mask)
            distribution = self.posterior_flow_model(params, self.flow_model)
        return distribution.sample(only_sample=only_sample)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        params = self.encoder(x, mask)
        return self.posterior_flow_model(params, self.flow_model)

class DiffusionModel(nn.Module):
    '''
        Model for using Diffusion Model as the posterior approximation
    '''
    def __init__(self, encoder: str, model_args, noise_coefficient = 1.):
        super(DiffusionModel, self).__init__()
        self.encoder = eval(encoder)(**model_args[encoder])
        self.score_wrapper = partial(ScoreWrapper, noise_coefficient=noise_coefficient)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        return self.score_wrapper(x, mask, self.encoder)

class FlowMatchingModel(nn.Module):
    '''
        Model for using Diffusion Model as the posterior approximation
    '''
    def __init__(self, encoder: str, model_args):
        super(FlowMatchingModel, self).__init__()
        self.encoder = eval(encoder)(**model_args[encoder])
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        return FlowWrapper(x, mask, self.encoder)

class PointModel(nn.Module):
    '''
        Model for using point estimates as the posterior approximation
    '''
    def __init__(self, encoder, model_args):
        super(PointModel, self).__init__()
        self.encoder = eval(encoder)(**model_args[encoder])

    def sample(self, x: torch.Tensor = None, mask: torch.Tensor = None, distribution = None):
        if distribution is None:
            return self(x, mask)
        return distribution
    
    def forward(self, x, mask):
        params = self.encoder(x, mask)
        return params