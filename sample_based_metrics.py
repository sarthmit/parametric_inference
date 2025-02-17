import os
from tqdm import tqdm
import wandb
import torch

# from data import InfiniteGaussianDataset
from posterior_utils import posterior_mean_full_gaussian
from distributions import GaussianDistribution, DiagonalGaussianDistribution

from discriminative_utils import Optimization, Langevin, HMC
from metrics import compute_distribution_distances
from plot_utils import plot_gaussian, plot_gaussian_separate
from utils import *
from args import get_args
from helpers import get_model_and_optimizer, get_dataset, get_experiment_name, \
                    get_params_dim, get_likelihood, get_evaluation_helpers
from trainer import *
from evaluator import *

device = torch.device('cuda')

@torch.no_grad()
def baseline(args, test_data, dataset, lengths):
    path = get_experiment_name(args)
    saved_path = f'results/{path}'
    eval_path = f'Baseline_SBM/{path}'
    os.makedirs(eval_path, exist_ok=True)
    in_dim, params_dim = get_params_dim(args)
    langevin = Langevin(params_dim, device)
    likelihood = get_likelihood(args.experiment, args.p_var)

    prior_args = {
        'dist': DiagonalGaussianDistribution,
        'mean': torch.zeros(params_dim).unsqueeze(0).to(device),
        'logvar': torch.zeros(params_dim).unsqueeze(0).to(device)
    }
    prior = prior_args['dist']((prior_args['mean'].repeat(num_test_datasets, 1),
                                prior_args['logvar'].repeat(num_test_datasets, 1)))

    with open(f'{eval_path}/log.txt', 'w') as f:
        for di, dt in enumerate(test_data):
            train_samples, val_samples, params, mask, params_mask = dt

            if isinstance(val_samples, tuple):
                train_x, train_y = train_samples
                x, y = val_samples
            else:
                train_x, train_y = train_samples, train_samples
                x, y = val_samples, val_samples

            langevin_samples, _, _ = langevin.chain(train_x, train_y, mask, params_mask, (params[1],), likelihood, dataset.conditional_fn, num_samples = 100, samples_per_chain = 25, show_tqdm=True)
            langevin_samples = langevin_samples.detach().cpu()
            prior_samples = torch.stack([prior.sample().detach().cpu() for _ in range(langevin_samples.shape[0])]).detach().cpu()

            prior_metrics = torch.zeros(5)
            for i in range(100):
                print(f'Evaluating Dataset: {i}')
                prior_metrics += compute_distribution_distances(prior_samples[:, i], langevin_samples[:, i]) / 100.
            
            f.write(f'Dimensionality: {lengths[di]} | W1: {prior_metrics[0]} | W2: {prior_metrics[1]} | Linear MMD: {prior_metrics[2]} | Poly MMD: {prior_metrics[3]} | RBF MMD: {prior_metrics[4]}\n')

@torch.no_grad()
def eval(args, test_data, dataset, lengths):
    path = get_experiment_name(args)
    saved_path = f'results/{path}'
    eval_path = f'Evaluation_SBM/{path}'

    os.makedirs(eval_path, exist_ok=True)
    in_dim, params_dim = get_params_dim(args)
    langevin = Langevin(params_dim, device)
    likelihood = get_likelihood(args.experiment, args.p_var)
    eval_func, ensemble_eval_func, eval_mode = get_evaluation_helpers(args.experiment, args.n_mixtures)

    model, optimizer = get_model_and_optimizer(args, in_dim, params_dim, device)

    '''
        Load from checkpoint if it exists and is readable
    '''
    print('Loading Checkpoint')
    state = torch.load(f'{saved_path}/model.pt')
    model.load_state_dict(state)

    with open(f'{eval_path}/log.txt', 'w') as f:
        for di, dt in enumerate(test_data):
            approx_posterior = approximate_posterior(dt, model, num_classes=args.num_classes)
            posterior_samples = None

            train_samples, val_samples, params, mask, params_mask = dt

            if isinstance(val_samples, tuple):
                train_x, train_y = train_samples
                x, y = val_samples
            else:
                train_x, train_y = train_samples, train_samples
                x, y = val_samples, val_samples

            langevin_samples, _, _ = langevin.chain(train_x, train_y, mask, params_mask, (params[1],), likelihood, dataset.conditional_fn, num_samples = 100, samples_per_chain = 25, show_tqdm=True)
            langevin_samples = langevin_samples.detach().cpu()
            approximate_samples = torch.stack([approx_posterior.sample().detach().cpu() for _ in range(langevin_samples.shape[0])]).detach().cpu()

            approximate_metrics = torch.zeros(5)
            for i in range(100):
                print(f'Evaluating Dataset: {i}')
                approximate_metrics += compute_distribution_distances(approximate_samples[:, i], langevin_samples[:, i]) / 100.
                
            f.write(f'Dimensionality: {lengths[di]} | W1: {approximate_metrics[0]} | W2: {approximate_metrics[1]} | Linear MMD: {approximate_metrics[2]} | Poly MMD: {approximate_metrics[3]} | RBF MMD: {approximate_metrics[4]}\n')

if __name__ == '__main__':
    args = get_args()
    num_test_datasets = 100

    if args.setup == 'fixed':
        lengths = [args.dim]
    else:
        if 'regression' in args.experiment:
            lengths = [1, args.dim // 2, args.dim]
        else:
            lengths = [2, args.dim // 2, args.dim]

    set_seed(0)
    dataset = get_dataset(args)
    test_data = [dataset.sample_batch(num_test_datasets, length) for length in lengths]

    set_seed(args.seed)
    if args.mode == 'baseline':
        baseline(args, test_data, dataset, lengths)
    else:
        eval(args, test_data, dataset, lengths)