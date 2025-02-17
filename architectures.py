import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import FrEIA.framework as Ff
import FrEIA.modules as Fm

import torch
import torch.nn as nn

class SinusoidalEmbedding(nn.Module):
    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        self.size = size
        self.scale = scale

    def forward(self, x: torch.Tensor):
        x = x * self.scale
        half_size = self.size // 2
        emb = torch.log(torch.Tensor([10000.0]).to(x.device)) / (half_size - 1)
        emb = torch.exp(-emb * torch.arange(half_size).to(x.device))
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return 0.01 * emb

    def __len__(self):
        return self.size

class Transformer(nn.Module):
    '''
        Transformer model as the set based architecture
    '''
    def __init__(self, in_dim: int, out_dim: int, dim: int, num_heads: int, num_layers: int):
        super(Transformer, self).__init__()

        self.in_dim = in_dim
        self.dim = dim
        self.out_dim = out_dim

        self.encoder = nn.Linear(in_dim, dim)

        tsf_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=4*dim, batch_first=False)
        self.model = nn.TransformerEncoder(tsf_layer, num_layers=num_layers)
        self.decoder = nn.Linear(dim, out_dim)
        self.state_encoder = nn.Linear(out_dim, dim)

        self.CLS = nn.Parameter(torch.zeros(1, 1, dim))
        self.time_embedding = SinusoidalEmbedding(dim)
        nn.init.xavier_uniform_(self.CLS)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, state: torch.Tensor = None, time: torch.Tensor = None):
        """
       :param x: with shape [S, N, D], S: Max length of Sets in Batch, N: Batch size, D: data dim
       :param mask: with shape [N, S], binary 0/1: indicates what elements to keep/delete respectively
       :return: output features with shape [N, D']
       """

        emb = self.encoder(x)
        if time is not None:
            state = self.state_encoder(state)
            time = self.time_embedding(time).unsqueeze(0)
            cls = state.unsqueeze(0) + time
        else:
            cls = self.CLS.repeat(1, x.shape[1], 1)

        emb = torch.cat([cls, emb], dim=0)

        if mask is not None:
            mask = torch.cat([torch.zeros_like(mask[:, :1]), mask], dim=1).bool()

        emb = self.model(src=emb, src_key_padding_mask=mask)
        return self.decoder(emb[0, :, :])
    
class GRU(nn.Module):
    def __init__(self, in_dim, out_dim, dim, num_layers):
        super(GRU, self).__init__()

        self.in_dim = in_dim
        self.dim = dim
        self.out_dim = out_dim

        self.encoder = nn.Linear(in_dim, dim)
        self.rnn = nn.GRU(dim, dim, num_layers=num_layers, batch_first=False, bidirectional=False)
        self.decoder = nn.Linear(dim, out_dim)

    def forward(self, x, mask):
        """
       :param x: with shape [S, N, D], S: Max length of Sets in Batch, N: Batch size, D: data dim
       :param mask: with shape [N, S], binary 0/1: indicates what elements to keep/delete respectively
       :return: output features with shape [N, D']
       """

        emb = self.encoder(x)
        emb = self.rnn(emb)[0]
        if mask is not None:
            mask_idx = torch.sum(1 - mask, dim=1) - 1
            outs = []
            for i in range(emb.shape[1]):
                outs.append(emb[mask_idx[i], i, :])
            outs = torch.stack(outs, dim=0)
        else:
            outs = emb[-1, :, :]

        return self.decoder(outs)


class DeepSets(nn.Module):
    # 4726276 parameters
    def __init__(self, in_dim: int, out_dim: int, dim: int, num_layers: int, param_gain: float = 2.45, aggregation: str = 'mean'):
        super(DeepSets, self).__init__()
        self.in_dim = in_dim
        self.embedding_features = int(dim * param_gain)
        self.inference_features = out_dim
        self.aggregation = aggregation

        embedding_layers = [nn.Linear(in_dim, self.embedding_features), nn.ReLU()]
        regression_layers = []

        for l in range(num_layers):
            embedding_layers.append(nn.Linear(self.embedding_features, self.embedding_features))
            embedding_layers.append(nn.ReLU())

            regression_layers.append(nn.Linear(self.embedding_features, self.embedding_features))
            regression_layers.append(nn.ReLU())

        regression_layers.append(nn.Linear(self.embedding_features, self.inference_features))

        self.embedding_network = nn.Sequential(
            *embedding_layers
        )
        self.regression_network = nn.Sequential(
            *regression_layers
        )
        self._init_weights(gain=.9)

    def _init_weights(self, gain: float):
        self.embedding_network.apply(lambda m: self._xavier_init(m, gain=gain))
        self.regression_network.apply(lambda m: self._xavier_init(m, gain=gain))

    def _xavier_init(self, m, gain: float = 1.):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0.)

    def _aggregation(self, x: torch.Tensor, mask: torch.Tensor):
        if self.aggregation == 'sum':
            x = torch.sum(x, dim=0)
        elif self.aggregation == 'max':
            x = torch.max(x, dim=0)
        else:
            if mask is not None:
                x = torch.sum(x, dim=0)
                N = torch.sum(mask, dim=0)
                x = torch.div(x, N)
            else:
                x = torch.mean(x, dim=0)
        return x

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        :param x: with shape [S, N, D], S: Max length of Sets in Batch, N: Batch size, D: data dim
        :param mask: with shape [N, S], binary 0/1: indicates what elements to keep/delete respectively
        :return: output features with shape [N, D']
        """
        # [S, N, D] -> [S, N, d]
        x = self.embedding_network(x)
        # [N, S] -> [S, N, 1]
        if mask is not None:
            mask = (1 - mask).t().unsqueeze(-1)
            x = x * mask
        # [S, N, d] -> [N, d]
        x = self._aggregation(x, mask)
        # [N, d] -> [N, D']
        x = self.regression_network(x)
        return x


class LinearTransformation(Fm.InvertibleModule):
    def __init__(self, dims_in, dims_c=[]):
        super().__init__(dims_in, dims_c)

    def forward(self, x, c, rev=False, jac=True, clamp=True):
        x, c = x[0], c[0]
        mean, logvar = torch.chunk(c, 2, dim=-1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        if not rev:
            x = mean + torch.exp(logvar*0.5) * x
            # forward operation
            log_jac_det = (logvar*0.5).sum(dim=-1)
        else:
            # backward operation
            x = (x - mean)/torch.exp(logvar*0.5)
            log_jac_det = -(logvar*0.5).sum(dim=-1)

        return (x,), log_jac_det

    def output_dims(self, input_dims):
        return input_dims


class SequentialNF(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_coupling):
        super().__init__()

        self.input_dims = (in_dim // 2,)
        self.cond_dims = (in_dim,)
        self.hidden_dim = hidden_dim

        # use the input_dims (2,) from above
        self.cinn = Ff.SequenceINN(*self.input_dims)
        self.cinn.append(LinearTransformation, cond=0, cond_shape=self.cond_dims)
        for k in range(num_coupling):
            # The cond=0 argument tells the operation which of the conditions it should
            # use, that are supplied with the call. So cond=0 means 'use the first condition'
            # (there is only one condition in this case).
            self.cinn.append(Fm.AllInOneBlock, cond=1, cond_shape=self.cond_dims, subnet_constructor=self._subnet_fc,
                             permute_soft=False)

    def _subnet_fc(self, dims_in, dims_out):
        """Return a feed-forward subnetwork, to be used in the coupling blocks below"""

        def f(dims_in, dims_out):
            net = nn.Sequential(nn.Linear(dims_in, self.hidden_dim), nn.ReLU(), nn.Dropout(p=0.01),
                                nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(), nn.Dropout(p=0.01),
                                nn.Linear(self.hidden_dim, dims_out))
            net.apply(lambda m: self._xavier_init(m, gain=1.))
            net[-1].weight.data.fill_(0.)
            return net

        return f(dims_in, dims_out)

    def _xavier_init(self, m, gain):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data, gain)

    def forward(self, z, h, rev=False):
        # the conditions have to be given as a list (in this example, a list with
        # one entry, 'one_hot_labels').  In general, multiple conditions can be
        # given. The cond argument of the append() method above specifies which
        # condition is used for each operation.
        theta, ldj = self.cinn(z, c=h, rev=rev)
        # ndim = z.shape[-1]
        return theta, ldj
