import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Amortized Estimators')
    '''
        Sampling Choices for Unsupervised Learning Data
    '''
    parser.add_argument('--p_var', type=str, default='identity', choices=('identity', 'diagonal', 'full'),
        help='variance style for data generating distribution p(x|\mu, \Sigma)')
    parser.add_argument('--sample_mixture_weights', action='store_true', default=False,
        help='Whether mixture components are sampled as well')
    parser.add_argument('--n_mixtures', type=int, default=2,
        help='Number of Mixture Components')

    '''
        Sampling Choices for Supervised Learning Data
    '''
    parser.add_argument('--sampling', type=str, default='normal', choices=('uniform', 'normal'),
        help='choice for sampling x')
    parser.add_argument('--num_classes', type=int, default=2,
        help='dataset size')
    parser.add_argument('--beta', type=float, default=0.25,
        help='variance in likelihood probability')
    parser.add_argument('--temperature', type=float, default=0.1,
        help='temperature in data generating process')

    '''
        Dataset Length Choices
    '''
    parser.add_argument('--min_len', type=int, default=64,
        help='minimum size of the dataset')
    parser.add_argument('--max_len', type=int, default=128,
        help='maximum size of the dataset')
    parser.add_argument('--min_param_len', type=int, default=1,
        help='minimum size of the dataset')
    parser.add_argument('--max_param_len', type=int, default=100,
        help='maximum size of the dataset')

    '''
        Data Generating Process Defaults
    '''
    parser.add_argument('--dim', type=int, default=2,
        help='dimensionality of the dataset')
    parser.add_argument('--h_dim', type=int, default=32,
        help='data generating hidden dim factor of the dataset')
    parser.add_argument('--d_layers', type=int, default=1,
        help='data generating layer factor of the dataset')
    parser.add_argument('--activation', type=str, default='tanh', choices=('tanh', 'relu'),
        help='activation function to use')

    '''
        Posterior Approximators
    '''
    parser.add_argument('--model', type=str, default='Vanilla', choices=('Vanilla', 'Flow'),
        help='Choose from ["Vanilla", "Flow"]')
    parser.add_argument('--nf_hidden', type=int, default=128,
        help='number of hidden dimensions for the subnetwork')
    parser.add_argument('--nf_cbs', type=int, default=6,
        help='number of coupling blocks of the normalizing flow')
    parser.add_argument('--noise_coefficient', type=float, default=1,
        help='noise coefficient for diffusion model')

    '''
        Backbone Encoding Models
    '''
    parser.add_argument('--encoder', type=str, default='Transformer', choices=('Transformer', 'DeepSets', 'GRU'),
        help='Choose from ["Transformer", "DeepSets"]')
    parser.add_argument('--hidden_dim', type=int, default=256,
        help='dimensionality for transformer')
    parser.add_argument('--heads', type=int, default=4,
        help='number of heads in transformer')
    parser.add_argument('--layers', type=int, default=4,
        help='number of layers in transformer/deepsets')
    parser.add_argument('--ds_aggregation', type=str, default='mean', choices=('sum', 'max', 'mean'),
        help='Aggregation operation used in DeepSets')

    '''
        Training Defaults
    '''
    parser.add_argument('--batch_size', type=int, default=128,
        help='batch size')
    parser.add_argument('--iters', type=int, default=10001,
        help='number of iterations')
    parser.add_argument('--warmup_iters', type=int, default=1,
        help='number of warmup iterations')
    parser.add_argument('--lr', type=float, default=1e-4,
        help='learning rate')

    '''
        Auxiliary Training Choices
    '''
    parser.add_argument('--mode', type=str, default='train', choices=('train', 'eval', 'baseline', 'true'),
        help='flag to define training vs evaluation')
    parser.add_argument('--setup', type=str, default='fixed', choices=('fixed', 'variable'),
        help='flag to define what training setup to use')
    parser.add_argument('--objective', type=str, default='backward', choices=('forward', 'backward', 'mixed', 'mle', 'map', 'diffusion', 'flow-matching', 'idem'),
        help='choice for the estimator objective')
    parser.add_argument('--train_data', type=str, default='linear', choices=('linear', 'mlp-tanh', 'mlp-relu', 'gp'),
        help='choice for training data for model misspecification case')

    '''
        Evaluation Defaults
    '''
    parser.add_argument('--eval_samples', type=int, default=25,
        help='Number of evaluation samples')

    '''
        Logging
    '''
    parser.add_argument('--wandb', action='store_true', default=False,
        help='flag to use wandb logging')
    parser.add_argument('--save_iters', type=int, default=10000,
        help='Number of intervals between consecutive savings and logging')

    '''
        Experiment Settings
    '''
    parser.add_argument('--experiment', type=str, default='gaussian', choices=('gaussian', 'gmm', 'linear_regression', 'linear_classification', 'nonlinear_regression', 'nonlinear_classification'),
        help='argument to decide which experiment to run')
    parser.add_argument('--seed', type=int, default=0,
        help='model and training seed')

    '''
        Tabular Dataset
    '''
    parser.add_argument('--dataset_idx', type=int, default=0,
        help='Tabular dataset index')

    args = parser.parse_args()
    return args