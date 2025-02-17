import wandb
import torch
import torch.utils.data as data_utils
from losses import *
from utils import one_hot, metrics_to_string
from trainer import preprocess_samples
from score_wrapper import ScoreWrapper, FlowWrapper

def compute_metrics(param_samples, x, y, mask, params_mask, extra_params, likelihood, conditional_fn, eval_func, ensemble_eval_func, name: str = 'Prior', evaluation_mode: str = 'Loss'):
    metrics = {
        f'{name} {evaluation_mode}': 0.,
        f'{name} Ensembled {evaluation_mode}': 0.,
        f'{name} CLL': 0.
    }

    for param_sample in param_samples:
        metrics[f'{name} {evaluation_mode}'] += \
            eval_func(param_sample, x,
            y, mask, params_mask, conditional_fn).item() / len(param_samples)
        metrics[f'{name} CLL'] += \
            conditional_log_likelihood(param_sample, x, y,
            mask, params_mask, extra_params, likelihood, conditional_fn).mean().item() / len(param_samples)

    '''
        Perform ensembled prediction; average if loss, mode if accuracy
    '''
    if ensemble_eval_func is None:
        return metrics

    y_pred = [
                conditional_fn(x, param_sample, params_mask) 
                    for param_sample in param_samples
            ]
    y_pred = torch.stack(y_pred)

    if evaluation_mode == "Accuracy":
        y_pred = torch.argmax(y_pred, dim=-1)
        y_pred = torch.mode(y_pred, dim=0).values
    else:
        y_pred = y_pred.mean(dim=0)

    metrics[f'{name} Ensembled {evaluation_mode}'] = \
        ensemble_eval_func(y_pred, y, mask)
    
    return metrics

def approximate_posterior(test_data, model, num_classes: int = None):
    model.eval()

    train_samples, val_samples, params, mask, params_mask = test_data
    _, _, samples = preprocess_samples(train_samples, num_classes)

    approx_posterior = model(samples, mask)
    return approx_posterior

def eval_step(test_data, model, conditional_fn, likelihood, eval_func, ensemble_eval_func, evaluation_mode, num_classes: int = None, it=None, eval_samples=25, log_wandb: bool = False, num_integration_steps = 100):
    model.eval()

    train_samples, val_samples, params, mask, params_mask = test_data
    _, _, samples = preprocess_samples(train_samples, num_classes)

    if isinstance(val_samples, tuple):
        x, y = val_samples
    else:
        x, y = val_samples, val_samples

    approx_posterior = model(samples, mask)

    if isinstance(approx_posterior, torch.Tensor):
        param_samples = [approx_posterior]
    elif isinstance(approx_posterior, ScoreWrapper) or isinstance(approx_posterior, FlowWrapper):
        param_samples = [
            approx_posterior.sample(num_steps=num_integration_steps) for _ in range(eval_samples)
        ]
    else:
        param_samples = [
            approx_posterior.sample() for _ in range(eval_samples)
        ]

    metrics = compute_metrics(param_samples, x, y, mask, params_mask, (params[1],), likelihood, conditional_fn, eval_func, ensemble_eval_func, name='Approximate', evaluation_mode=evaluation_mode)

    if it is not None:
        if log_wandb:
            wandb.log(metrics, step=it)
    
    return approx_posterior, metrics_to_string(metrics)