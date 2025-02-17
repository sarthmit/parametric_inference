import random
import numpy as np
import torch
import torch.nn.functional as F

def set_seed(seed):
    '''
        Set Seed for Reproducibility
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def dcp(tensor):
    '''
        Detach and set to CPU
    '''
    return tensor.detach().cpu()

def dcpn(tensor):
    '''
        Detach, set to CPU and convert to numpy
    '''
    return dcp(tensor).numpy()

def check_psd(value):
    '''
        Check if value is postive semi-definite
    '''
    return torch.distributions.constraints.positive_definite.check(value).all().item()

def one_hot(tensor, num_classes):
    '''
        One Hot Encoding
    '''
    shape = tensor.size()[:-1]
    return F.one_hot(tensor.view(-1), num_classes).view(shape + (num_classes,))

def metrics_to_string(metrics):
    '''
        Convert metrics dictionary to string
    '''
    log = f''
    for key in sorted(metrics.keys()):
        log += f'{key}: {metrics[key]} | '
    log = log[:-2] + '\n'
    return log
