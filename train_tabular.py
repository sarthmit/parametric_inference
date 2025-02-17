import os
from tqdm import tqdm
import wandb
import torch
import pickle
from itertools import cycle
from sklearn.model_selection import KFold
import pandas as pd

from posterior_utils import posterior_mean_full_gaussian
from distributions import GaussianDistribution, DiagonalGaussianDistribution

from score_wrapper import ScoreWrapper, FlowWrapper
from discriminative_utils import Optimization, Langevin, HMC
from plot_utils import plot_gaussian, plot_gaussian_separate
from utils import *
from args import get_args
from helpers import get_model_and_optimizer, get_dataset, get_experiment_name, \
                    get_params_dim, get_likelihood, get_evaluation_helpers
from trainer import *
from evaluator import *

device = torch.device('cuda')

def get_model(args, param_size):
    if args.experiment == 'linear_regression':
        model = nn.Linear(param_size, 1).to(device)
    elif args.experiment == 'linear_classification':
        model = nn.Linear(param_size, 2).to(device)
    elif args.experiment == 'nonlinear_regression':
        if args.activation == 'relu':
            model = nn.Sequential(
                nn.Linear(param_size, 32),
                nn.ReLU(),
                nn.Linear(32, 1, bias=False)
            ).to(device)
        else:
            model = nn.Sequential(
                nn.Linear(param_size, 32),
                nn.Tanh(),
                nn.Linear(32, 1, bias=False)
            ).to(device)
    elif args.experiment == 'nonlinear_classification':
        if args.activation == 'relu':
            model = nn.Sequential(
                nn.Linear(param_size, 32),
                nn.ReLU(),
                nn.Linear(32, 2, bias=False)
            ).to(device)
        else:
            model = nn.Sequential(
                nn.Linear(param_size, 32),
                nn.Tanh(),
                nn.Linear(32, 2, bias=False)
            ).to(device)
    
    return model

def initialize(model, parameter, param_size):
    if parameter is None:
        return model

    if args.experiment == 'linear_regression':
        model.weight.data = parameter[:, :param_size]
        model.bias.data = parameter[0, 100]
    elif args.experiment == 'linear_classification':
        parameter = parameter.view(101, 2).transpose(1, 0)
        model.weight.data = parameter[:, :param_size]
        model.bias.data = parameter[:, 100]
    elif args.experiment == 'nonlinear_regression':
        model[0].weight.data = parameter[0, :param_size * 32].reshape(param_size, 32).T
        model[0].weight.bias = parameter[0, 100 * 32: 101 * 32]
        model[2].weight.data = parameter[0, 101 * 32:].reshape(32, 1).T
    elif args.experiment == 'nonlinear_classification':
        model[0].weight.data = parameter[0, :param_size * 32].reshape(param_size, 32).T
        model[0].weight.bias = parameter[0, 100 * 32: 101 * 32]
        model[2].weight.data = parameter[0, 101 * 32:].reshape(32, 2).T
    
    return model

def get_padded_input(X_train, X_test):
    pad_mask = torch.ones(1, 100).to(device)
    pad_mask[:, :X_train.shape[1]] = 0
    X_pad = torch.zeros(X_train.shape[0] + X_test.shape[0], 100).to(X_train.device)
    X_pad[:X_train.shape[0], :X_train.shape[1]] = X_train
    X_pad[X_train.shape[0]:, :X_test.shape[1]] = X_test

    X_train, X_test = X_pad[:X_train.shape[0]], X_pad[X_train.shape[0]:]
    return X_train, X_test, pad_mask

def normalize(X_train, y_train, X_test, y_test, regression=False):
  x_mean = X_train.mean(dim=0, keepdim=True)
  x_std = X_train.std(dim=0, keepdim=True)
  X_train = (X_train - x_mean) / (x_std + 1e-8)
  X_test = (X_test - x_mean) / (x_std + 1e-8)

  if regression:
    y_mean = y_train.mean(dim=0, keepdim=True)
    y_std = y_train.std(dim=0, keepdim=True) / (X_train.shape[-1]**0.5)
    y_train = (y_train - y_mean) / (y_std + 1e-8)
    y_test = (y_test - y_mean) / (y_std + 1e-8)

  return X_train, y_train, X_test, y_test

def get_data(dataset_idx, regression=True):
    if regression:
        f = open(f'data/regression_datasets.pickle', 'rb')
    else:
        f = open(f'data/classification_datasets.pickle', 'rb')
    
    datasets = pickle.load(f)
    dataset = datasets[dataset_idx]
    kfold = cycle(KFold(n_splits=5, shuffle=True, random_state=0).split(dataset[1]))
    return dataset, kfold

def l2_loss(logits, y):
    return (logits - y).pow(2).mean()

def bce_loss(logits, y):
    dist = torch.distributions.categorical.Categorical(logits=logits)
    return -dist.log_prob(y).mean()

def accuracy(logits, y):
    y_pred = logits.argmax(dim=-1).unsqueeze(-1)
    acc = (y_pred == y).float()
    return 100. * acc.mean()

def optimization(test_data, model, parameter, loss_fn, metric_fn, param_size):
    '''
        Returns a dataframe of the optimization process
    '''
    X_train, y_train, X_test, y_test = test_data
    initialize(model, parameter, param_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1.)
    test_metrics = []

    for i in range(2501):
        if i % 10 == 0:
            model.eval()
            logits = model(X_test)
            metric = metric_fn(logits, y_test)
            test_metrics.append(metric.item())

        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        train_loss = loss_fn(logits, y_train)
        train_loss.backward()
        optimizer.step()
    
    return np.array(test_metrics)

def kfold_optimization(dataset_idx, param_sampler, experiment, eval_idx, name):
    if 'classification' in experiment:
        regression = False
        label_preprocess = lambda y: y.long()
        loss_fn = bce_loss
        metric_fn = accuracy
    else:
        regression = True
        label_preprocess = lambda y: y.float()
        loss_fn = l2_loss
        metric_fn = l2_loss

    dataset, kfold = get_data(dataset_idx, regression)
    X, y = torch.Tensor(dataset[1]).to(device), label_preprocess(torch.Tensor(dataset[2]).to(device)).unsqueeze(-1)
    params_dim = X.shape[-1]

    metrics = []
    for fold_idx in range(5):
        train_index, test_index = next(kfold)
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        X_train, y_train, X_test, y_test = normalize(X_train, y_train, X_test, y_test, regression=regression)

        downstream_model = get_model(args, params_dim)
        if param_sampler is not None:
            padded_X_train, padded_X_test, params_mask = get_padded_input(X_train, X_test)
            parameter = param_sampler(padded_X_train, y_train)
        else:
            parameter = None
        
        test_data = (X_train, y_train, X_test, y_test)

        test_metric = optimization(test_data, downstream_model, parameter, loss_fn, metric_fn, params_dim)
        metrics.append(test_metric)
    
    metrics = np.mean(np.stack(metrics, axis=0), axis=0)
    iters = [i for i in range(0, 2501, 10)]
    eval_list = [eval_idx for _ in range(len(iters))]
    seed_list = [args.seed for _ in range(len(iters))]
    name_list = [name for _ in range(len(iters))]
    dataset_list = [dataset[0] for _ in range(len(iters))]

    dataframe = np.stack([name_list, dataset_list, eval_list, seed_list, iters, metrics], axis=-1)
    dataframe = pd.DataFrame(dataframe, columns=['Experiment Name', 'Dataset', 'Evaluation Id', 'Seed', 'Iteration', 'Performance'])
    dataframe = dataframe.astype({'Experiment Name': str, 'Dataset': str, 'Evaluation Id': int, 'Seed': int, 'Iteration': int, 'Performance': float})
    return dataframe

def run_optimization(args, dataset_idx, sampler, name):
    dfs = []
    for eval_idx in range(args.eval_samples):
        print(f'Evaluation: {eval_idx}')
        dfs.append(kfold_optimization(dataset_idx, sampler, args.experiment, eval_idx, name))
    df = pd.concat(dfs, ignore_index=True)
    print(df)
    return df

def run_inference(args, dataset_idx, sampler, name):
    in_dim, params_dim = get_params_dim(args)
    dataset = get_dataset(args)
    conditional_fn = dataset.conditional_fn
    likelihood = get_likelihood(args.experiment, args.p_var)
    eval_func, ensemble_eval_func, eval_mode = get_evaluation_helpers(args.experiment)

    if 'classification' in args.experiment:
        regression = False
        label_preprocess = lambda y: y.long()
    else:
        regression = True
        label_preprocess = lambda y: y.float()

    dataset, kfold = get_data(dataset_idx, regression)
    X, y = torch.Tensor(dataset[1]).to(device), label_preprocess(torch.Tensor(dataset[2]).to(device)).unsqueeze(-1)
    metrics = []

    for fold_idx in range(5):
        train_index, test_index = next(kfold)
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        X_train, y_train, X_test, y_test = normalize(X_train, y_train, X_test, y_test, regression=regression)
        X_train, X_test, params_mask = get_padded_input(X_train, X_test)

        parameters = [sampler(X_train, y_train) for _ in range(args.eval_samples)]
        metric = compute_metrics(parameters, X_test.unsqueeze(1), y_test.unsqueeze(1), None, params_mask, (None,), likelihood, conditional_fn, eval_func, ensemble_eval_func, name=name, evaluation_mode=eval_mode)
        metric = metrics_to_string(metric)
        metrics.append(metric)
    
    return metrics

def baseline(args):
    dataset_idx = args.dataset_idx
    path = get_experiment_name(args)
    eval_path = f'Baseline_tabular/{path}'
    os.makedirs(eval_path, exist_ok=True)
    in_dim, params_dim = get_params_dim(args)

    prior_args = {
        'dist': DiagonalGaussianDistribution,
        'mean': torch.zeros(params_dim).unsqueeze(0).to(device),
        'logvar': torch.zeros(params_dim).unsqueeze(0).to(device)
    }
    prior_dist = prior_args['dist']((prior_args['mean'].repeat(1, 1),
                                prior_args['logvar'].repeat(1, 1)))

    metrics = run_inference(args, dataset_idx, lambda X, y: prior_dist.sample(), 'Prior')
    with open(f'{eval_path}/{dataset_idx}.txt', 'w') as f:
        for fold_idx, metric in enumerate(metrics):
            f.write(f'Fold Index: {fold_idx} | {metric}')

    df_normal = run_optimization(args, dataset_idx, lambda X, y: prior_dist.sample(), 'Normal')
    df_normal.to_pickle(f'{eval_path}/{dataset_idx}_prior.pkl')
    df_xavier = run_optimization(args, dataset_idx, None, 'Xavier')
    df_xavier.to_pickle(f'{eval_path}/{dataset_idx}_xavier.pkl')

def eval(args):
    dataset_idx = args.dataset_idx
    path = get_experiment_name(args)
    saved_path = f'results/{path}'
    eval_path = f'Evaluation_tabular/{path}'
    os.makedirs(eval_path, exist_ok=True)
    in_dim, params_dim = get_params_dim(args)

    model, optimizer = get_model_and_optimizer(args, in_dim, params_dim, device)
    '''
        Load from checkpoint if it exists and is readable
    '''
    print('Loading Checkpoint')
    state = torch.load(f'{saved_path}/model.pt')
    model.load_state_dict(state)

    def sampler(X_train, y_train):
        test_data = ((X_train.unsqueeze(1), y_train.unsqueeze(1)), None, None, None, None)
        approx_posterior = approximate_posterior(test_data, model, num_classes=args.num_classes)
        if isinstance(approx_posterior, torch.Tensor):
            return approx_posterior
        elif isinstance(approx_posterior, ScoreWrapper) or isinstance(approx_posterior, FlowWrapper):
            return approx_posterior.sample(num_steps = 100)
        else:
            return approx_posterior.sample()

    metrics = run_inference(args, dataset_idx, sampler, 'Approximate')
    with open(f'{eval_path}/{dataset_idx}.txt', 'w') as f:
        for fold_idx, metric in enumerate(metrics):
            f.write(f'Fold Index: {fold_idx} | {metric}')

    df = run_optimization(args, dataset_idx, sampler, f'{args.objective}_{args.encoder}')
    df.to_pickle(f'{eval_path}/{dataset_idx}_pretrained.pkl')

if __name__ == '__main__':
    args = get_args()
    if args.mode == 'eval':
        set_seed(args.seed)
        eval(args)
    elif args.mode == 'baseline':
        set_seed(args.seed)
        baseline(args)