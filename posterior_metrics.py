import os
from tqdm import tqdm
import wandb
import torch

# from data import InfiniteGaussianDataset
from posterior_utils import posterior_mean_full_gaussian
from distributions import GaussianDistribution, DiagonalGaussianDistribution

from discriminative_utils import Optimization, Langevin, HMC
from plot_utils import plot_gaussian, plot_gaussian_separate
from utils import *
from args import get_args
from helpers import get_model_and_optimizer, get_dataset, get_experiment_name, \
                    get_params_dim, get_likelihood, get_evaluation_helpers
from trainer import *
from evaluator import *
from posterior_utils import *

device = torch.device('cuda')

def true_posterior_metric():
    pass

@torch.no_grad()
def eval(args, test_data, dataset, lengths):
    sym_kl = []
    for seed in range(3):
        args.seed = seed
        path = get_experiment_name(args)
        saved_path = f'results/{path}'
        in_dim, params_dim = get_params_dim(args)
        model, optimizer = get_model_and_optimizer(args, in_dim, params_dim, device)

        '''
            Load from checkpoint if it exists and is readable
        '''
        print('Loading Checkpoint')
        state = torch.load(f'{saved_path}/model.pt')
        model.load_state_dict(state)

        prior_args = {
            'dist': DiagonalGaussianDistribution,
            'mean': torch.zeros(params_dim).unsqueeze(0).to(device),
            'logvar': torch.zeros(params_dim).unsqueeze(0).to(device)
        }
        prior = prior_args['dist']((prior_args['mean'].repeat(num_test_datasets, 1),
                                    prior_args['logvar'].repeat(num_test_datasets, 1)))

        for di, dt in enumerate(test_data):
            approx_posterior = approximate_posterior(dt, model, num_classes=args.num_classes)

            if args.experiment == 'linear_regression':
                (x, y), _, params, mask, params_mask = dt
                posterior = posterior_linear_regression(x, y, mask, args.beta, device=device)
            elif args.experiment == 'gaussian':
                samples, _, params, mask, params_mask = dt
                posterior = posterior_mean_full_gaussian(samples, mask, params[1], device=device)
            
            sym_kl.append(0.5 * (kl_normal_normal(posterior, approx_posterior) + \
                    kl_normal_normal(approx_posterior, posterior)
                    ).mean().item() / params_dim)
    
    string = f'${np.mean(sym_kl):.3f}$\\sstd' + '{' + f'${np.std(sym_kl):.3f}$' + '}'
    with open('log.txt', 'a') as f:
        f.write(f'{string}\n')

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
    eval(args, test_data, dataset, lengths)