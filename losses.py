import torch
import torch.nn as nn
from tqdm import tqdm
from data import *

EPS=1e-8

def cond_loss(y_: torch.Tensor, y: torch.Tensor, mask: torch.Tensor):
    '''
        Computes the loss between prediction y_ and targets y
    '''
    loss = (y_ - y) ** 2
    if mask is not None:
        loss = (loss * (1 - mask).transpose(1, 0).unsqueeze(-1)).sum(dim=-1).sum(dim=0)
        count = (1 - mask).sum(dim=1)
        loss = loss / count
    else:
        loss = loss.sum(dim=-1).mean(dim=0)

    return loss.mean()

def cond_acc(y_: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, logits: bool = True):
    '''
        Computes the accuracy with logits y_ and targets y
    '''
    if logits:
        params = torch.argmax(y_, dim=-1)
    else:
        params = y_
    
    acc = torch.eq(params, y.squeeze(-1))
    if mask is not None:
        acc = (acc * (1 - mask).transpose(1, 0)).sum(dim=0)
        count = (1 - mask).sum(dim=1)
        acc = acc / count
    else:
        acc = acc.float().mean(dim=0)
    return (100. * acc).mean()

def conditional_loss(params: torch.Tensor, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, params_mask: torch.Tensor, conditional_fn=None):
    '''
        Computes the predictiive loss with certain model parameters, inputs and targets
    '''
    params = conditional_fn(x, params, params_mask)
    return cond_loss(params, y, mask)

def conditional_accuracy(params: torch.Tensor, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, params_mask: torch.Tensor, conditional_fn=None):
    '''
        Computes the predictiive accuracy with certain model parameters, inputs and targets
    '''
    params = conditional_fn(x, params, params_mask)
    return cond_acc(params, y, mask)

def conditional_log_likelihood(params: torch.Tensor, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, params_mask: torch.Tensor, extra_params=None, likelihood=None, conditional_fn=None):
    '''
        Computes the log likelihood of the model with certain parameters, inputs and targets
        extra_params consist of parameters not explicitly modeled with amortized inference, eg. variance of prediction
        likelihood defines the distribution family that defines the likelihood function
    '''
    params = conditional_fn(x, params, params_mask)
    if extra_params[0] is not None:
        params = (params, ) + extra_params

    likelihood = likelihood(params)
    log_prob = likelihood.nll(y, params_mask)
    if mask is not None:
        log_prob = log_prob * (1-mask).transpose(1,0)
    return log_prob.sum(dim=0)

def conditional_matching_loss(params: torch.Tensor, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, params_mask: torch.Tensor, conditional_fn=None, n_mixtures: int = 2):
    '''
        Computes best assignment for each point to cluster and returns resulting loss
    '''
    batch_size = params.shape[0]
    params = conditional_fn(x, params, params_mask).view(batch_size, n_mixtures, -1)
    loss = ((params.unsqueeze(0) - x.unsqueeze(-2)) ** 2)
    loss = loss.min(dim=-2)[0]
    loss = loss.sum(dim=-1) * (1 - mask).transpose(1, 0)
    loss = loss.sum(dim=0)
    count = (1 - mask).sum(dim=1)
    return (loss / count).mean()

def kl_normal_normal(normal1, normal2):
    '''
        Computes KL(normal1 || normal2) for arbitrary Gaussian
    '''

    mean1, var1 = normal1.mean, normal1.cov
    mean2, var2 = normal2.mean, normal2.cov

    var2_inv = torch.pinverse(var2)
    dim = mean1.shape[-1]

    log_det_var_1 = torch.logdet(var1)
    log_det_var_2 = torch.logdet(var2)

    kl = log_det_var_2 - log_det_var_1 - \
        dim + \
        torch.diagonal(torch.matmul(var2_inv, var1), dim1=1, dim2=2).sum(dim=-1) + \
        ((mean2 - mean1) * torch.matmul(var2_inv, (mean2 - mean1).unsqueeze(-1)).squeeze(-1)).sum(dim=-1)

    return 0.5 * kl

def kl_logvar(normal1, normal2):
    '''
        Computes KL(normal1 || normal2) for diagonal densities
    '''

    mean1, logvar1 = normal1.mean, normal1.logvar
    mean2, logvar2 = normal2.mean, normal2.logvar

    kl = (logvar2 - logvar1 - 1 + logvar1.exp() / logvar2.exp() + \
        (mean2 - mean1) ** 2 / logvar2.exp()).sum(dim=-1)

    return 0.5 * kl

# TODO: Set two sample classifier test more cleanly
def two_sample(train_dataloader, test_dataloader, in_dim, device, epochs = 100):
  def accuracy(logits, y):
    logits = (logits >= 0).int()
    return (logits == y).sum().item()

  model = nn.Sequential(
    nn.Linear(in_dim, 512),
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, 1)
  ).to(device)
  model.train()

  optim = torch.optim.Adam(model.parameters(), lr=1e-3)
  criterion = nn.BCEWithLogitsLoss()

  iterator = tqdm(range(epochs), unit="#Epochs", ncols=100, leave=True) 

  for _ in iterator:
    acc = 0.
    count = 0.
    total_loss = 0.
    for x,y in train_dataloader:
      x, y = x.to(device), y.to(device)
      optim.zero_grad()
      out = model(x).squeeze(-1)
      loss = criterion(out, y)
      acc += accuracy(out, y)
      total_loss += loss.item() * x.shape[0]
      count += x.shape[0]
      loss.backward()
      optim.step()
    
    iterator.set_description(f"Loss: {(total_loss / count):.3f} Accuracy: {(acc / count) * 100.:.3f}")
  
  acc = 0.
  count = 0
  for x,y in test_dataloader:
      x, y = x.to(device), y.to(device)
      out = model(x).squeeze(-1)
      acc += accuracy(out, y)
      count += x.shape[0]

  del model, optim
  return acc / count  

if __name__ == '__main__':
  mean1 = torch.randn(32, 2)
  mean2 = torch.randn(32, 2)
  var1 = torch.randn(32, 2).exp()
  var2 = torch.randn(32, 2).exp()

  g1 = DiagonalGaussianDistribution(torch.cat([mean1, torch.log(var1)], dim=-1), trunc=False)
  g2 = DiagonalGaussianDistribution(torch.cat([mean2, torch.log(var2)], dim=-1), trunc=False)

  print(g1.kl(other=g2))
  print(kl_normal_normal((mean1, torch.diag_embed(var1)), (mean2, torch.diag_embed(var2))))