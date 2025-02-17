import numpy as np
import matplotlib.pyplot as plt
import torch
from utils import dcp

def plot_gaussian(args, data, approx_posterior, name, s=10):
  if args.dim != 2:
    return

  (_, samples, params, masks) = data

  for idx in range(10):
    data = samples[:, idx]
    mask = masks[idx]
    data = data[:(1-mask).sum().int().item()]

    true_mean = params[0][idx]

    approx_samples = torch.cat(
      [
        approx_posterior.sample()[idx].unsqueeze(0) for _ in range(args.eval_samples)
       ], dim=0
    )

    plt.scatter(dcp(approx_samples[:, 0]), dcp(approx_samples[:, 1]),
                c='red', s=s, marker='^', alpha=0.5)
    plt.scatter(dcp(data[:, 0]), dcp(data[:, 1]),
                c='blue', s=s)
    plt.scatter(dcp(true_mean[0]), dcp(true_mean[1]), marker='*', c='black', s=2*s)
    plt.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)
    plt.savefig(name+f'{idx}.png', bbox_inches='tight')
    plt.savefig(name+f'{idx}.pdf', bbox_inches='tight')
    plt.close()

def plot_gaussian_separate(args, data, approx_posterior, true_posterior, optimized_params, mcmc_params, it, name, s=10):
  if args.dim != 2:
    return

  (_, samples, params, masks) = data
  figures = 5
  if mcmc_params is None:
    figures = 4

  for idx in range(10):
    fig, axs = plt.subplots(1, figures, figsize=(20, figures))
    data = samples[:, idx]
    mask = masks[idx]
    data = data[:(1-mask).sum().int().item()]
    true_mean = params[0][idx]

    posterior_samples = torch.cat(
      [
        true_posterior.sample()[idx].unsqueeze(0) for _ in range(args.eval_samples)
      ], dim=0)
    approx_samples = torch.cat(
      [
        approx_posterior.sample()[idx].unsqueeze(0) for _ in range(args.eval_samples)
       ], dim=0
    )

    for i in range(figures):
      axs[i].scatter(dcp(data[:, 0]), dcp(data[:, 1]), c='blue', s=s)

    axs[0].scatter(dcp(true_mean[0]), dcp(true_mean[1]), c='black', s=s)

    axs[1].scatter(dcp(posterior_samples[:, 0]), dcp(posterior_samples[:, 1]),
                c='green', s=s, marker='*')

    axs[2].scatter(dcp(approx_samples[:, 0]), dcp(approx_samples[:, 1]),
                c='red', s=s, marker='*')

    axs[3].scatter(dcp(optimized_params[:, idx, 0]), dcp(optimized_params[:, idx, 1]),
                c='cyan', s=s, marker='*')

    if mcmc_params is not None:
      axs[4].scatter(dcp(mcmc_params[:, idx, 0]), dcp(mcmc_params[:, idx, 1]),
                c='purple', s=s, marker='*')

    plt.savefig(f'{name}/Plots/{idx}_{it}.png',
                bbox_inches='tight')
    plt.close()

def plot_regression(args, data, approx_posterior, name, conditional_fn, s=10, alpha=0.25):
  if args.dim != 1:
    return

  (_, samples, params, masks) = data
  xs, ys = samples
  x_linear = torch.linspace(xs.min(), xs.max(), steps=32).unsqueeze(-1).unsqueeze(1).to(xs.device)

  for idx in range(10):
    x, y = xs[:, idx], ys[:, idx]
    mask = masks[idx]
    x = x[:(1 - mask).sum().int().item()]
    y = y[:(1 - mask).sum().int().item()]

    for _ in range(args.eval_samples):
        pred = conditional_fn(x_linear, approx_posterior.sample()[idx].unsqueeze(0))
        plt.plot(dcp(x_linear[:, 0, 0]), dcp(pred[:, 0, 0]),
                  c='red', alpha=alpha)

    plt.scatter(dcp(x[:, 0]), dcp(y[:, 0]), c='blue', s=s)

    pred = conditional_fn(x_linear, params[0][idx].unsqueeze(0))
    plt.plot(dcp(x_linear[:, 0, 0]), dcp(pred[:, 0, 0]), c='black')
    plt.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)
    plt.savefig(name + f'{idx}.png', bbox_inches='tight')
    plt.savefig(name + f'{idx}.pdf', bbox_inches='tight')
    plt.close()

def plot_regression_separate(args, data, approx_posterior, true_posterior, optimized_params, 
                             mcmc_params, it, name, conditional_fn, s=5, alpha=0.25):
  if args.dim != 1:
    return

  num_frames = 5
  additive = 1
  if true_posterior is None:
    num_frames -= 1
    additive = 0
  if mcmc_params is None:
    num_frames -= 1

  (_, samples, params, masks) = data
  xs, ys = samples
  x_linear = torch.linspace(xs.min(), xs.max(), steps=32).unsqueeze(-1).unsqueeze(1).to(xs.device)

  for idx in range(10):
    fig, axs = plt.subplots(1, num_frames, figsize=(num_frames * 5, num_frames))

    x, y = xs[:, idx], ys[:, idx]
    mask = masks[idx]
    x = x[:(1 - mask).sum().int().item()]
    y = y[:(1 - mask).sum().int().item()]

    for i in range(num_frames):
      axs[i].scatter(dcp(x[:, 0]), dcp(y[:, 0]), c='blue', s=s)

    pred = conditional_fn(x_linear, params[0][idx].unsqueeze(0))
    axs[0].plot(dcp(x_linear[:, 0, 0]), dcp(pred[:, 0, 0]), c='black', label='gt w')

    for i in range(args.eval_samples):
      if true_posterior is not None:
        pred = conditional_fn(x_linear, true_posterior.sample()[idx].unsqueeze(0))
        axs[additive].plot(dcp(x_linear[:, 0, 0]), dcp(pred[:, 0, 0]),
                    c='green', alpha=alpha)

      pred = conditional_fn(x_linear, approx_posterior.sample()[idx].unsqueeze(0))
      axs[1+additive].plot(dcp(x_linear[:, 0, 0]), dcp(pred[:, 0, 0]),
                  c='red', alpha=alpha)

      pred = conditional_fn(x_linear, optimized_params[i, idx].unsqueeze(0))
      axs[2+additive].plot(dcp(x_linear[:, 0, 0]), dcp(pred[:, 0, 0]),
                  c='cyan', alpha=alpha)

      if mcmc_params is not None:
        pred = conditional_fn(x_linear, mcmc_params[i, idx].unsqueeze(0))
        axs[3+additive].plot(dcp(x_linear[:, 0, 0]), dcp(pred[:, 0, 0]),
                  c='purple', alpha=alpha)

    fig.savefig(f'{name}/Plots/{idx}_{it}.png',
                bbox_inches='tight')
    plt.close()

def make_grid(X):
  device = X.device
  X = X.detach().cpu().numpy()
  min1, max1 = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
  min2, max2 = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

  x1grid = np.arange(min1, max1, 0.025)
  x2grid = np.arange(min2, max2, 0.025)

  xx, yy = np.meshgrid(x1grid, x2grid)

  r1, r2 = xx.flatten(), yy.flatten()
  r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))

  grid = np.hstack((r1,r2))

  return (xx, yy), torch.tensor(grid).float().to(device)

def plot_classification_contour(x, y, params, axis, conditional_fn, gt=False, temperature=0.1, s=10, eval_samples=25):
  (xx, yy), grid = make_grid(x)
  levels = np.linspace(0., 1., 10)

  if gt:
    pr = conditional_fn(grid.unsqueeze(1), params.unsqueeze(0))
    prob = torch.softmax(pr / temperature, dim=-1)[:, 0, 0]
  else:
    pred = []
    for p in range(eval_samples):
      pr = conditional_fn(grid.unsqueeze(1), params[p].unsqueeze(0))
      prob = torch.softmax(pr, dim=-1)[:, :, 0]
      pred.append(prob)
    prob = torch.cat(pred, dim=1).mean(dim=1)
  prob = prob.view(xx.shape)
  c = axis.contourf(xx, yy, prob.detach().cpu().numpy(), cmap='RdBu', levels=levels, vmax=1., vmin=0.)
  axis.scatter(dcp(x[:, 0]), dcp(x[:, 1]), c=dcp(y[:, 0]), s=s, cmap='Paired')
  axis.tick_params(left=False, right=False, labelleft=False,
                  labelbottom=False, bottom=False)
  return c

def plot_classification_separate(args, data, approx_posterior, optimized_params, mcmc_params, it, name, conditional_fn, s=25):
  if args.dim != 2:
    return

  (_, samples, params, masks) = data
  xs, ys = samples
  approx_posterior_samples = torch.cat([approx_posterior.sample().unsqueeze(0) for _ in range(args.eval_samples)], dim=0)
  num_frames = 4
  if mcmc_params is None:
    num_frames -= 1

  for idx in range(10):
    fig, axs = plt.subplots(1, num_frames, figsize=(num_frames * 5, num_frames))

    x, y = xs[:, idx], ys[:, idx]
    mask = masks[idx]
    x = x[:(1 - mask).sum().int().item()]
    y = y[:(1 - mask).sum().int().item()]

    plot_classification_contour(x, y, params[0][idx], axs[0], conditional_fn, gt=True, temperature=0.1, eval_samples=args.eval_samples)
    plot_classification_contour(x, y, approx_posterior_samples[:, idx], axs[1], conditional_fn, gt=False, eval_samples=args.eval_samples)
    c = plot_classification_contour(x, y, optimized_params[:, idx], axs[2], conditional_fn, gt=False, eval_samples=args.eval_samples)
    if mcmc_params is not None:
      c = plot_classification_contour(x, y, mcmc_params[:, idx], axs[3], conditional_fn, gt=False, eval_samples=args.eval_samples)

    plt.colorbar(c)
    fig.savefig(f'{name}/Plots/{idx}_{it}.png',
                bbox_inches='tight')
    plt.close()

def plot_classification(args, data, approx_posterior, name, conditional_fn, s=10):
  if args.dim != 2:
    return

  (_, samples, params, masks) = data
  xs, ys = samples

  for idx in range(10):
    x, y = xs[:, idx], ys[:, idx]
    mask = masks[idx]
    x = x[:(1 - mask).sum().int().item()]
    y = y[:(1 - mask).sum().int().item()]

    (xx, yy), grid = make_grid(x)

    pred = []
    for p in range(args.eval_samples):
      pr = conditional_fn(grid.unsqueeze(1), approx_posterior.sample()[idx].unsqueeze(0))
      prob = torch.softmax(pr, dim=-1)[:, :, 0]
      pred.append(prob)
    pred = torch.cat(pred, dim=1).mean(dim=1)
    pred = pred.view(xx.shape)
    c = plt.contourf(xx, yy, pred.detach().cpu().numpy(), cmap='RdBu')

    pred = []
    for p in range(args.eval_samples):
      pr = conditional_fn(x.unsqueeze(1), approx_posterior.sample()[idx].unsqueeze(0))
      pr = torch.max(pr, dim=-1)[1]
      pred.append(pr)
    pred = torch.cat(pred, dim=1).mode(dim=1)[0]
    plt.scatter(dcp(x[:, 0]), dcp(x[:, 1]), c=dcp(pred), s=s, cmap='Paired')

    # plt.colorbar(c)
    plt.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)
    plt.savefig(name + f'{idx}.png', bbox_inches='tight')
    plt.savefig(name+f'{idx}.pdf', bbox_inches='tight')
    plt.close()

def plot_gmm(args, data, approx_posterior, name, s=10):
  if args.dim != 2:
    return

  (_, samples, params, masks) = data
  bsz = samples.shape[1]
  mean = params[0].view(bsz, args.n_mixtures, args.dim)
  approx_means = [approx_posterior.sample().unsqueeze(1) for _ in range(args.eval_samples)]
  approx_means = torch.cat(approx_means, dim=1).view(bsz, args.eval_samples, args.n_mixtures, args.dim)

  for idx in range(10):
    x = samples[:, idx]
    mask = masks[idx]
    x = x[:(1 - mask).sum().int().item()]

    plt.scatter(dcp(x[:, 0]), dcp(x[:, 1]), c='blue', s=s)
    plt.scatter(dcp(approx_means[idx, :, 0, 0]), dcp(approx_means[idx, :, 0, 1]), c='red', marker='^', alpha=0.5, s=s)
    plt.scatter(dcp(approx_means[idx, :, 1, 0]), dcp(approx_means[idx, :, 1, 1]), c='red', marker='^', alpha=0.5, s=s)
    plt.scatter(dcp(mean[idx, :, 0]), dcp(mean[idx, :, 1]), c='black', marker='*', s=2*s)
    plt.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)

    plt.savefig(name + f'{idx}.png', bbox_inches='tight')
    plt.savefig(name + f'{idx}.pdf', bbox_inches='tight')
    plt.close()

def plot_gmm_separate(args, data, approx_posterior, optimized_params, mcmc_params, it, name):
  if args.dim != 2:
    return
  
  figures = 5
  if mcmc_params is None:
    figures -= 1

  (_, samples, params, masks) = data
  bsz = samples.shape[1]
  mean = params[0].view(bsz, args.n_mixtures, args.dim)
  approx_means = [approx_posterior.sample().unsqueeze(1) for _ in range(args.eval_samples)]
  approx_means = torch.cat(approx_means, dim=1).view(bsz, args.eval_samples, args.n_mixtures, args.dim)
  op_params = optimized_params.view(args.eval_samples, bsz, args.n_mixtures, args.dim)
  mcmc_params = mcmc_params.view(args.eval_samples, bsz, args.n_mixtures, args.dim)

  for idx in range(10):
    fig, axs = plt.subplots(1, figures, figsize=(figures * 5, figures))

    x = samples[:, idx]
    mask = masks[idx]
    x = x[:(1 - mask).sum().int().item()]

    for i in range(figures):
      axs[i].scatter(dcp(x[:, 0]), dcp(x[:, 1]))

    axs[0].scatter(dcp(mean[idx, :, 0]), dcp(mean[idx, :, 1]), c='black', marker='*')
    axs[1].scatter(dcp(approx_means[idx, :, :, 0]), dcp(approx_means[idx, :, :, 1]), c='red', marker='*')
    axs[2].scatter(dcp(approx_means[idx, :, 0, 0]), dcp(approx_means[idx, :, 0, 1]), c='green', marker='*')
    axs[2].scatter(dcp(approx_means[idx, :, 1, 0]), dcp(approx_means[idx, :, 1, 1]), c='brown', marker='*')
    axs[3].scatter(dcp(op_params[:, idx, :, 0]), dcp(op_params[:, idx, :, 1]), c='cyan', marker='*')
    if mcmc_params is not None:
      axs[4].scatter(dcp(mcmc_params[:, idx, :, 0]), dcp(mcmc_params[:, idx, :, 1]), c='purple', marker='*')

    plt.savefig(f'{name}/Plots/{idx}_{it}.png',
                bbox_inches='tight')
    plt.close()

def plot_gmm_trial(args, data, approx_posterior, name, s=40):
  if args.dim != 2:
    return

  (_, samples, params, masks) = data
  bsz = samples.shape[1]
  mean = params[0].view(bsz, args.n_mixtures, args.dim)
  approx_means = [approx_posterior.sample().unsqueeze(1) for _ in range(args.eval_samples)]
  approx_means = torch.cat(approx_means, dim=1).view(bsz, args.eval_samples, args.n_mixtures, args.dim)

  for idx in range(10):
    x = samples[:, idx]
    mask = masks[idx]
    x = x[:(1 - mask).sum().int().item()]

    plt.scatter(dcp(x[:, 0]), dcp(x[:, 1]), c='blue', s=20)
    plt.scatter(dcp(approx_means[idx, :, 0, 0]), dcp(approx_means[idx, :, 0, 1]), c='green', marker='^', alpha=0.5, s=s)
    plt.scatter(dcp(approx_means[idx, :, 1, 0]), dcp(approx_means[idx, :, 1, 1]), c='brown', marker='^', alpha=0.5, s=s)
    plt.scatter(dcp(mean[idx, :, 0]), dcp(mean[idx, :, 1]), c='black', marker='*', s=2*s)
    plt.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)

    plt.savefig(name + f'{idx}.png', bbox_inches='tight')
    plt.savefig(name + f'{idx}.pdf', bbox_inches='tight')
    plt.close()
