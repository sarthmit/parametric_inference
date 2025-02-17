import torch
import numpy as np

def euler_maruyama_sde_step(score, noise_schedule, t, x, dt):
    diffusion = noise_schedule.g(t)
    drift = score(torch.zeros(x.shape[0]).to(x.device) + t, x)

    # Update the state
    x_next = x + drift * dt + diffusion * torch.randn_like(x) * (dt ** 0.5)
    return x_next

def euler_maruyama_ode_step(drift, t, x, dt):
    dx = drift(torch.zeros(x.shape[0]).to(x.device) + t, x)
    x_next = x - dx * dt
    return x_next

def integrate_sde(
    score,
    noise_schedule,
    x1,
    num_integration_steps
):
    x = x1.clone()
    times = torch.linspace(1., 0., num_integration_steps + 1, device=x.device)[:-1]
    with torch.no_grad():
        for t in times:
            x = euler_maruyama_sde_step(
                score, noise_schedule, t, x, 1. / num_integration_steps
            )

    return x

def integrate_ode(
    drift,
    x1,
    num_integration_steps
):
    x = x1.clone()
    times = torch.linspace(1., 0., num_integration_steps + 1, device=x.device)[:-1]
    with torch.no_grad():
        for t in times:
            x = euler_maruyama_ode_step(
                drift, t, x, 1. / num_integration_steps
            )

    return x

class QuadNoiseSchedule:
    def __init__(self, beta):
        self.beta = beta

    def f(self, t):
        return torch.zeros_like(t) + self.beta

    def g(self, t):
        return (2 * t * (self.beta ** 2)) ** 0.5

    def h(self, t):
        return (self.beta * t) ** 2

class LinearNoiseSchedule:
    def __init__(self, beta):
        self.beta = beta

    def g(self, t):
        return torch.full_like(t, self.beta**0.5)

    def h(self, t):
        return self.beta * t


class GeometricNoiseSchedule:
    def __init__(self, sigma_min, sigma_max):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_diff = self.sigma_max / self.sigma_min

    def g(self, t):
        return (
            self.sigma_min
            * (self.sigma_diff**t)
            * ((2 * np.log(self.sigma_diff)) ** 0.5)
        )

    def h(self, t):
        return (self.sigma_min * (((self.sigma_diff ** (2 * t)) - 1) ** 0.5)) ** 2

class ScoreWrapper():
    def __init__(self, samples, mask, model, noise_coefficient=1.):
        self.noise_schedule = QuadNoiseSchedule(noise_coefficient ** 0.5)
        self.samples = samples
        self.mask = mask
        self.model = model
        self.dim = self.model.out_dim
    
    def score(self, t, state):
        return self.model(self.samples, self.mask, state, t)

    def sample(self, size=None, num_steps = 100):
        std = self.noise_schedule.h(1) ** 0.5
        if size is None:
            size = []
            size = torch.Size(size)
        
        z = std * torch.randn(size + (self.samples.shape[1], self.dim,)).to(self.samples.device)
        return integrate_sde(self.score, self.noise_schedule, z, num_steps)

class FlowWrapper():
    def __init__(self, samples, mask, model):
        self.samples = samples
        self.mask = mask
        self.model = model
        self.dim = self.model.out_dim
    
    def drift(self, t, state):
        return self.model(self.samples, self.mask, state, t)

    def sample(self, size=None, num_steps = 100):
        if size is None:
            size = []
            size = torch.Size(size)
        
        z = torch.randn(size + (self.samples.shape[1], self.dim,)).to(self.samples.device)
        return integrate_ode(self.drift, z, num_steps)