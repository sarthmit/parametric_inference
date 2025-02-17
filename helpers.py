import torch
from distributions import *
from models import GaussianModel, FlowModel, PointModel, DiffusionModel, FlowMatchingModel
from data import *
from losses import *
from functools import partial

def get_experiment_name(args):
    '''
        Get the wandb name of the experiment as well as the path
    '''
    path = f'{args.setup}/{args.objective}/{args.experiment}/{args.model}_{args.encoder}_{args.dim}'

    if 'nonlinear' in args.experiment:
        path += f'_{args.activation}_{args.d_layers}'

    if args.experiment == 'gmm':
        path += f'_{args.n_mixtures}'
    elif 'classification' in args.experiment:
        path += f'_{args.num_classes}'
    
    if args.objective == 'diffusion' or args.objective == 'idem':
        path += f'_{args.noise_coefficient}'
    
    path += f'/{args.seed}'
    return path

def get_params_dim(args):
    if args.experiment == 'gaussian':
        return args.dim, args.dim
    elif args.experiment == 'gmm':
        return args.dim, args.dim * args.n_mixtures
    elif args.experiment == 'linear_regression':
        return args.dim + 1, args.dim + 1
    elif args.experiment == 'linear_classification':
        return args.dim + args.num_classes, (args.dim + 1) * args.num_classes
    elif args.experiment == 'nonlinear_regression':
        return args.dim + 1, (args.dim + 1) * args.h_dim + \
               (args.h_dim * args.h_dim * (args.d_layers - 1)) + \
               args.h_dim
    elif args.experiment == 'nonlinear_classification':
        return args.dim + args.num_classes, (args.dim + 1) * args.h_dim + args.h_dim * args.num_classes + \
               args.h_dim * args.h_dim * (args.d_layers - 1)
    else:
        raise NotImplementedError

def get_model_and_optimizer(args, in_dim: int, out_dim: int, device):
    '''
        Returns the amortization model as well as the optimizer
    '''
    if args.objective == 'mle' or args.objective == 'map' or args.objective == 'diffusion' or args.objective == 'idem' or args.objective == 'flow-matching':
        out_multiplier = 1
    else:
        out_multiplier = 2

    if args.model == 'Flow':
        gain = 2
    else:
        gain = 1

    model_args = {
        "Transformer": {
                'in_dim': in_dim,
                'dim': args.hidden_dim,
                'out_dim': out_multiplier * out_dim * gain,
                'num_heads': args.heads,
                'num_layers': args.layers
            },
        "GRU": {
                'in_dim': in_dim,
                'dim': args.hidden_dim,
                'out_dim': out_multiplier * out_dim * gain,
                'num_layers': args.layers
            },
        "DeepSets": {
                'in_dim': in_dim,
                'dim': args.hidden_dim,
                'out_dim': out_multiplier * out_dim * gain,
                'num_layers': args.layers,
                'aggregation': args.ds_aggregation
            },
        "Posterior_Flow": {
                'in_dim': out_multiplier * out_dim,
                'hidden_dim': args.nf_hidden,
                'num_coupling': args.nf_cbs,
            },
        }
    
    if args.objective == 'mle' or args.objective == 'map':
        model = PointModel(encoder=args.encoder, model_args=model_args).to(device)
    elif args.objective == 'diffusion' or args.objective == 'idem':
        model = DiffusionModel(encoder=args.encoder, model_args=model_args, noise_coefficient=args.noise_coefficient).to(device)
    elif args.objective == 'flow-matching':
        model = FlowMatchingModel(encoder=args.encoder, model_args=model_args).to(device)
    elif args.model == 'Flow':
        model = FlowModel(encoder=args.encoder, model_args=model_args).to(device)
    else:
        model = GaussianModel(encoder=args.encoder, model_args=model_args).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(model)
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of Parameters: {model_params}')
    return model, optimizer

def get_dataset(args):
    '''
        Get the dataset for the experiment
    '''
    device = torch.device('cuda')
    if args.setup == 'fixed':
        args.min_param_len = args.dim
        args.max_param_len = args.dim

    if args.activation == 'tanh':
        act = torch.tanh
    elif args.activation == 'relu':
        act = nn.ReLU()

    if args.experiment == 'gaussian':
        return Gaussian(dim=args.dim, min_len=args.min_len, max_len=args.max_len,
                        min_param_len=args.min_param_len, max_param_len=args.max_param_len,
                        variance=args.p_var, mode=args.setup, device=device)
    elif args.experiment == 'gmm':
        return GMM(dim=args.dim, n_mixtures=args.n_mixtures, min_len=args.min_len, max_len=args.max_len,
                   min_param_len=args.min_param_len, max_param_len=args.max_param_len, variance=args.p_var, 
                   sample_mixture_weights=args.sample_mixture_weights, mode=args.setup, device=device)
    elif args.experiment == 'linear_regression':
        return LinearRegression(dim=args.dim, min_len=args.min_len, max_len=args.max_len,
                                min_param_len=args.min_param_len, max_param_len=args.max_param_len,
                                beta=args.beta, x_sampling=args.sampling, mode=args.setup, device=device)
    elif args.experiment == 'linear_classification':
        return LinearClassification(dim=args.dim, num_classes=args.num_classes, min_len=args.min_len, max_len=args.max_len,
                                    min_param_len=args.min_param_len, max_param_len=args.max_param_len,
                                    temperature=args.temperature, x_sampling=args.sampling, mode=args.setup, device=device)
    elif args.experiment == 'nonlinear_regression':
        return NonLinearRegression(dim=args.dim, h_dim=args.h_dim, n_layers=args.d_layers, act=act, x_sampling=args.sampling,
                                   min_len=args.min_len, max_len=args.max_len, min_param_len=args.min_param_len,
                                   max_param_len=args.max_param_len, mode=args.setup, beta=args.beta, device=device)
    elif args.experiment == 'nonlinear_classification':
        return NonLinearClassification(dim=args.dim, h_dim=args.h_dim, n_layers=args.d_layers, act=act, x_sampling=args.sampling,
                                       num_classes=args.num_classes, min_len=args.min_len, max_len=args.max_len,
                                       min_param_len=args.min_param_len, max_param_len=args.max_param_len,
                                       temperature=args.temperature, mode=args.setup, device=device)
    elif args.experiment == 'gp':
        return GP(dim=args.dim, min_len=args.min_len, max_len=args.max_len, min_param_len=args.min_param_len, 
                  max_param_len=args.max_param_len, x_sampling=args.sampling, mode=args.setup, device=device)
    else:
        raise NotImplementedError

def get_likelihood(experiment, p_var):
    '''
        Get the likelihood for the experiment
    '''
    if experiment == 'gaussian':
        if p_var == 'full':
            return partial(GaussianDistribution, unsupervised=True)
        else:
            return partial(DiagonalGaussianDistribution, unsupervised=True)

    elif experiment == 'gmm':
        if p_var == 'full':
            return partial(MixtureGaussianDistribution, unsupervised=True)
        else:
            return partial(MixtureDiagonalGaussianDistribution, unsupervised=True)

    elif 'classification' in experiment:
        return CategoricalDistribution
    
    elif 'regression' in experiment:
        return DiagonalGaussianDistribution

def get_evaluation_helpers(experiment, n_mixtures: int = 2):
    if 'classification' in experiment:
        eval_func = conditional_accuracy
        ensemble_func = partial(cond_acc, logits=False)
        eval_mode = 'Accuracy'
    elif 'gmm' in experiment:
        eval_func = partial(conditional_matching_loss, n_mixtures=n_mixtures)
        ensemble_func = None
        eval_mode = 'Matching Loss'
    else:
        eval_func = conditional_loss
        ensemble_func = cond_loss
        eval_mode = 'Loss'

    return eval_func, ensemble_func, eval_mode

def get_misspecification_data(args):
    datasets = []

    # Get dataset functions
    experiment = args.experiment
    act = args.activation

    args.experiment = 'linear_regression'
    datasets.append(get_dataset(args))

    args.experiment = 'gp'
    datasets.append(get_dataset(args))

    args.experiment = 'nonlinear_regression'
    args.activation = 'tanh'
    datasets.append(get_dataset(args))

    args.activation = 'relu'
    datasets.append(get_dataset(args))

    if args.train_data == 'linear':
        train_dataset = datasets[0]    
    elif args.train_data == 'gp':
        train_dataset = datasets[1]
    elif args.train_data == 'mlp-tanh':
        train_dataset = datasets[2]
    elif args.train_data == 'mlp-relu':
        train_dataset = datasets[3]

    args.experiment = experiment
    args.activation = act

    return datasets, train_dataset, ['linear', 'gp', 'mlp-tanh', 'mlp-relu']