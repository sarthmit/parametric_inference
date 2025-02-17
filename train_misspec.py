import os
from tqdm import tqdm
import wandb
import torch

from posterior_utils import posterior_mean_full_gaussian
from distributions import GaussianDistribution, DiagonalGaussianDistribution

from discriminative_utils import Optimization, Langevin, HMC
from plot_utils import plot_gaussian, plot_gaussian_separate
from utils import *
from args import get_args
from helpers import get_model_and_optimizer, get_dataset, get_experiment_name, \
                    get_params_dim, get_likelihood, get_evaluation_helpers, get_misspecification_data
from trainer import *
from evaluator import *

device = torch.device('cuda')

def train(args, test_data, dataset, conditional_fn, names):
    path = get_experiment_name(args)
    path = f'results_misspec/{args.train_data}/{path}'
    os.makedirs(path, exist_ok=True)
    in_dim, params_dim = get_params_dim(args)
    likelihood = get_likelihood(args.experiment, args.p_var)
    eval_func, ensemble_eval_func, eval_mode = get_evaluation_helpers(args.experiment)
    noise_schedule = QuadNoiseSchedule(args.noise_coefficient)

    config = args.__dict__
    config["Experiment"] = f'{args.setup}-{args.experiment}'
    if args.wandb:
        wandb.init(project="Amortized_Inference", entity="sarthmit",
                config=config, name=path)

    prior_args = {
        'dist': DiagonalGaussianDistribution,
        'mean': torch.zeros(params_dim).unsqueeze(0).to(device),
        'logvar': torch.zeros(params_dim).unsqueeze(0).to(device)
    }

    model, optimizer = get_model_and_optimizer(args, in_dim, params_dim, device)

    '''
        Load from checkpoint if it exists and is readable
    '''
    if os.path.exists(f'{path}/model_last.pt'):
        try:
            print('Trying to load model')
            state = torch.load(f'{path}/model_last.pt')
            model.load_state_dict(state['state_dict'])
            start_iteration = state['iteration']
        except:
            print('Failed to load model')
            start_iteration = 0
    else:
        start_iteration = 0

    tqdm_batch = tqdm(range(start_iteration, args.iters), unit="batch", ncols=100, leave=True)

    with open(f'{path}/log.txt', 'w') as f:
        for it in tqdm_batch:
            loss = train_step(model, optimizer, dataset, conditional_fn, likelihood, prior_args, args, noise_schedule, it)
            tqdm_batch.set_postfix({"loss": loss})

            '''
                Log evaluation metrics and save a checkpoint every 10,000 iterations
            '''
            if it % args.save_iters == 0:
                f.write(f'Iteration: {it} | Training Loss: {loss}\n')
                for di, dt in enumerate(test_data):
                    approx_posterior, test_metrics = eval_step(dt, model, conditional_fn, likelihood,
                                            eval_func, ensemble_eval_func, eval_mode, 
                                            num_classes=args.num_classes, it=it, 
                                            eval_samples=args.eval_samples, log_wandb=args.wandb,
                                            num_integration_steps = 100)
                    f.write(f'Iteration: {it} | Eval Data: {names[di]} | {test_metrics}')

                torch.save({
                        'state_dict': model.state_dict(),
                        'iteration': it
                    }, f'{path}/model_last.pt')

    torch.save(model.state_dict(), f'{path}/model.pt')

def prior_metric(eval_path, prior, test_data, names, likelihood, conditional_fn, eval_func, ensemble_eval_func, eval_mode, num_eval_samples):
    '''
        Get prior metrics for the baseline methods
    '''

    with open(f'{eval_path}/prior.txt', 'w') as f:
        for di, dt in enumerate(test_data):
            train_samples, val_samples, params, mask, params_mask = dt

            if isinstance(val_samples, tuple):
                x, y = val_samples
            else:
                x, y = val_samples, val_samples

            parameters = [prior.sample() for _ in range(num_eval_samples)]
            prior_metrics = compute_metrics(parameters, x, y, mask, params_mask, (params[1],), likelihood, conditional_fn, eval_func, ensemble_eval_func, name='Prior', evaluation_mode=eval_mode)
            prior_metrics = metrics_to_string(prior_metrics)
            # parameters = torch.stack([prior.sample() for _ in range(20000)])
            # torch.save(parameters, f'{eval_path}/prior-parameters_{names[di]}.pt')
            f.write(f'Eval Data: {names[di]} | {prior_metrics}')

def optimization_metric(eval_path, optimizer, test_data, names, likelihood, conditional_fn, eval_func, ensemble_eval_func, eval_mode, num_eval_samples):
    '''
        Get optimization metrics for the baseline methods
    '''
    with open(f'{eval_path}/optimization.txt', 'w') as f:
        for di, dt in enumerate(test_data):
            train_samples, val_samples, params, mask, params_mask = dt

            if isinstance(val_samples, tuple):
                train_x, train_y = train_samples
                x, y = val_samples
            else:
                train_x, train_y = train_samples, train_samples
                x, y = val_samples, val_samples

            parameters = optimizer.train(train_x, train_y, mask, params_mask, (params[1],), likelihood, conditional_fn, num_samples=num_eval_samples, show_tqdm=True)
            # torch.save(parameters, f'{eval_path}/optimization-parameters_{names[di]}.pt')

            parameters = parameters[:num_eval_samples]
            optimization_metrics = compute_metrics(parameters, x, y, mask, params_mask, (params[1],), likelihood, conditional_fn, eval_func, ensemble_eval_func, name='Optimization', evaluation_mode=eval_mode)
            optimization_metrics = metrics_to_string(optimization_metrics)

            f.write(f'Eval Data: {names[di]} | {optimization_metrics}')

def langevin_metric(eval_path, langevin, test_data, names, likelihood, conditional_fn, eval_func, ensemble_eval_func, eval_mode, num_eval_samples):
    '''
        Get langevin metrics for the baseline methods
    '''
    with open(f'{eval_path}/langevin.txt', 'w') as f:
        for di, dt in enumerate(test_data):
            train_samples, val_samples, params, mask, params_mask = dt

            if isinstance(val_samples, tuple):
                train_x, train_y = train_samples
                x, y = val_samples
            else:
                train_x, train_y = train_samples, train_samples
                x, y = val_samples, val_samples

            parameters, parameters_last, parameters_single = langevin.chain(train_x, train_y, mask, params_mask, (params[1],), likelihood, conditional_fn, num_samples = num_eval_samples, show_tqdm=True)
            # torch.save(parameters, f'{eval_path}/langevin-parameters_{names[di]}.pt')

            eval_parameters = parameters_last[:num_eval_samples]
            langevin_metrics = compute_metrics(eval_parameters, x, y, mask, params_mask, (params[1],), likelihood, conditional_fn, eval_func, ensemble_eval_func, name='Langevin Chain', evaluation_mode=eval_mode)
            langevin_metrics = metrics_to_string(langevin_metrics)
            f.write(f'Eval Data: {names[di]} | {langevin_metrics}')

            eval_parameters = parameters_single[-num_eval_samples:]
            langevin_metrics = compute_metrics(eval_parameters, x, y, mask, params_mask, (params[1],), likelihood, conditional_fn, eval_func, ensemble_eval_func, name='Langevin Single', evaluation_mode=eval_mode)
            langevin_metrics = metrics_to_string(langevin_metrics)
            f.write(f'Eval Data: {names[di]} | {langevin_metrics}')

def hmc_metric(eval_path, hmc, test_data, names, likelihood, conditional_fn, eval_func, ensemble_eval_func, eval_mode, num_eval_samples):
    '''
        Get langevin metrics for the baseline methods
    '''
    with open(f'{eval_path}/hmc.txt', 'w') as f:
        for di, dt in enumerate(test_data):
            train_samples, val_samples, params, mask, params_mask = dt

            if isinstance(val_samples, tuple):
                train_x, train_y = train_samples
                x, y = val_samples
            else:
                train_x, train_y = train_samples, train_samples
                x, y = val_samples, val_samples

            parameters, parameters_last, parameters_single = hmc.chain(train_x, train_y, mask, params_mask, (params[1],), likelihood, conditional_fn, num_samples = num_eval_samples, show_tqdm=True)
            # torch.save(parameters, f'{eval_path}/hmc-parameters_{names[di]}.pt')

            eval_parameters = parameters_last[:num_eval_samples]
            langevin_metrics = compute_metrics(eval_parameters, x, y, mask, params_mask, (params[1],), likelihood, conditional_fn, eval_func, ensemble_eval_func, name='HMC Chain', evaluation_mode=eval_mode)
            langevin_metrics = metrics_to_string(langevin_metrics)
            f.write(f'Eval Data: {names[di]} | {langevin_metrics}')

            eval_parameters = parameters_single[-num_eval_samples:]
            langevin_metrics = compute_metrics(eval_parameters, x, y, mask, params_mask, (params[1],), likelihood, conditional_fn, eval_func, ensemble_eval_func, name='HMC Single', evaluation_mode=eval_mode)
            langevin_metrics = metrics_to_string(langevin_metrics)
            f.write(f'Eval Data: {names[di]} | {langevin_metrics}')

def true_posterior_metric():
    pass

def baseline(args, test_data, dataset, conditional_fn, names, num_test_datasets = 100):
    path = get_experiment_name(args)
    eval_path = f'Baseline_misspec/{args.train_data}/{path}'
    os.makedirs(eval_path, exist_ok=True)
    in_dim, params_dim = get_params_dim(args)
    likelihood = get_likelihood(args.experiment, args.p_var)
    eval_func, ensemble_eval_func, eval_mode = get_evaluation_helpers(args.experiment)

    prior_args = {
        'dist': DiagonalGaussianDistribution,
        'mean': torch.zeros(params_dim).unsqueeze(0).to(device),
        'logvar': torch.zeros(params_dim).unsqueeze(0).to(device)
    }
    prior = prior_args['dist']((prior_args['mean'].repeat(num_test_datasets, 1),
                                prior_args['logvar'].repeat(num_test_datasets, 1)))
    prior_metric(eval_path, prior, test_data, names, likelihood, conditional_fn, eval_func, ensemble_eval_func, eval_mode, args.eval_samples)

    optimizer = Optimization(params_dim, device)
    optimization_metric(eval_path, optimizer, test_data, names, likelihood, conditional_fn, eval_func, ensemble_eval_func, eval_mode, args.eval_samples)

    langevin = Langevin(params_dim, device)
    langevin_metric(eval_path, langevin, test_data, names, likelihood, conditional_fn, eval_func, ensemble_eval_func, eval_mode, args.eval_samples)

    hmc = HMC(params_dim, device)
    hmc_metric(eval_path, hmc, test_data, names, likelihood, conditional_fn, eval_func, ensemble_eval_func, eval_mode, args.eval_samples)

def eval(args, test_data, dataset, conditional_fn, names):
    path = get_experiment_name(args)
    saved_path = f'results_misspec/{args.train_data}/{path}'
    eval_path = f'Evaluation_misspec/{args.train_data}/{path}'

    os.makedirs(eval_path, exist_ok=True)
    in_dim, params_dim = get_params_dim(args)
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
            approx_posterior, test_metrics = eval_step(dt, model, conditional_fn, likelihood,
                                    eval_func, ensemble_eval_func, eval_mode, num_classes=args.num_classes,
                                    it=None, eval_samples=args.eval_samples, log_wandb=args.wandb,
                                    num_integration_steps = 100)
            f.write(f'Eval Data: {names[di]} | {test_metrics}')

if __name__ == '__main__':
    args = get_args()
    num_test_datasets = 100
    assert(args.setup == 'fixed')

    datasets, train_dataset, exp_names = get_misspecification_data(args)
    conditional_fn = get_dataset(args).conditional_fn
    test_data = []
    for dataset in datasets:
        set_seed(0)
        test_data.append(dataset.sample_batch(num_test_datasets, args.dim))

    if args.mode == 'eval':
        set_seed(args.seed)
        eval(args, test_data, train_dataset, conditional_fn, exp_names)
    elif args.mode == 'baseline':
        set_seed(args.seed)
        baseline(args, test_data, train_dataset, conditional_fn, exp_names)
    elif args.mode == 'train':
        if "SLURM_PROCID" in os.environ:
            args.seed += int(os.environ["SLURM_PROCID"])
            num_threads = int(os.environ.get('SLURM_CPUS_PER_TASK','1'))
            torch.set_num_threads(num_threads)
        set_seed(args.seed)
        train(args, test_data, train_dataset, conditional_fn, exp_names)