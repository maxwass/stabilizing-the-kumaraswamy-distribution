import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

#from torch_geometric.nn.models import InnerProductDecoder, VGAE
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn.inits import reset
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

import matplotlib.pyplot as plt

from typing import Optional, Tuple

from kumaraswamy import KumaraswamyStable
from simple_squashed_normal import TanhNormal

EPS = 1e-15
MAX_LOGSTD = 10


""" 
    Much of this is taken from:
        https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/autoencoder.html#InnerProductDecoder
"""

class InnerProductDecoder(torch.nn.Module):
    r"""The inner product decoder from the `"Variational Graph Auto-Encoders"
    <https://arxiv.org/abs/1611.07308>`_ paper.

    .. math::
        \sigma(\mathbf{Z}\mathbf{Z}^{\top})

    where :math:`\mathbf{Z} \in \mathbb{R}^{N \times d}` denotes the latent
    space produced by the encoder.
    """
    def forward(
        self,
        z: Tensor,
        edge_index: Tensor,
        sigmoid: bool = True,
    ) -> Tensor:
        r"""Decodes the latent variables :obj:`z` into edge probabilities for
        the given node-pairs :obj:`edge_index`.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            edge_index (torch.Tensor): The edge indices.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z: Tensor, sigmoid: bool = True) -> Tensor:
        r"""Decodes the latent variables :obj:`z` into a probabilistic dense
        adjacency matrix.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj

class MLPDecoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(MLPDecoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_channels, 1)
        )

    def forward(
        self,
        z: Tensor,
        edge_index: Tensor,
        sigmoid: bool = True,
    ) -> Tensor:
        r"""Decodes the latent variables :obj:`z` into edge probabilities for
        the given node-pairs :obj:`edge_index`.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            edge_index (torch.Tensor): The edge indices.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        value = self.mlp(z[edge_index[0]] * z[edge_index[1]]).squeeze()
        return torch.sigmoid(value) if sigmoid else value

class VariationalMLPDecoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_distrib):
        super(VariationalMLPDecoder, self).__init__()
        self.mlp_1 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_channels, 1)
        )
        self.mlp_2 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_channels, 1)
        )
        self.latent_distrib = latent_distrib
    
    def forward(
        self,
        z: Tensor,
        edge_index: Tensor,
        #sigmoid: bool = True,
    ) -> Tensor:
        r"""Decodes the latent variables :obj:`z` into a KS or Beta 
        distribution modeling the edge probabilities for
        the given node-pairs :obj:`edge_index`.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            edge_index (torch.Tensor): The edge indices.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        param_1 = self.mlp_1(z[edge_index[0]] * z[edge_index[1]]).squeeze()
        param_2 = self.mlp_2(z[edge_index[0]] * z[edge_index[1]]).squeeze()
        if self.latent_distrib == 'ks':
            log_a, log_b = param_1, param_2
            distrib = KumaraswamyStable(log_a, log_b)
        elif self.latent_distrib == 'beta':
            a, b = F.softplus(param_1) + 1e-6, F.softplus(param_2) + 1e-6
            #a, b = torch.exp(log_a), torch.exp(log_b)
            distrib =  torch.distributions.Beta(a, b)
        elif self.latent_distrib == 'tanh-normal':
            mu, log_stdv = param_1, param_2
            distrib = TanhNormal(mu, log_stdv, low=0, high=1)
        else:
            raise ValueError(f"Invalid latent distribution: {self.latent_distrib}")
        return distrib

class GAE(torch.nn.Module):
    r"""The Graph Auto-Encoder model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper based on user-defined encoder and decoder models.

    Args:
        encoder (torch.nn.Module): The encoder module.
        decoder (torch.nn.Module, optional): The decoder module. If set to
            :obj:`None`, will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """
    def __init__(self, encoder: Module, decoder: Module):
        super().__init__()
        self.encoder = encoder
        #self.decoder = InnerProductDecoder() if decoder is None else decoder
        self.decoder = decoder
        GAE.reset_parameters(self)

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        reset(self.encoder)
        reset(self.decoder)

    def forward(self, *args, **kwargs) -> Tensor:  # pragma: no cover
        r"""Alias for :meth:`encode`."""
        return self.encoder(*args, **kwargs)

    def encode(self, *args, **kwargs) -> Tensor:
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs) -> Tensor:
        r"""Runs the decoder and computes edge probabilities."""
        return self.decoder(*args, **kwargs)
    
    def recon_loss(self, z: Tensor, pos_edge_index: Tensor, neg_edge_index) -> Tensor:
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.

            Args:
                z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
                pos_edge_index (torch.Tensor): The positive edges to train against.
                neg_edge_index (torch.Tensor): The negative edges to
                    train against.
        """
    
        pos_loss = -torch.log(self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + EPS).mean()

        return pos_loss + neg_loss

    def test(self, e: Tensor, pos_edge_index: Tensor,
             neg_edge_index: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Given nodal embeddings :obj:`e`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes area under the ROC curve (AUC) and average precision (AP)
        scores.

        Args:
            e (torch.Tensor): The final nodal embeddings:math:`\mathbf{E}`.
            pos_edge_index (torch.Tensor): The positive edges to evaluate
                against.
            neg_edge_index (torch.Tensor): The negative edges to evaluate
                against.
        """
        pos_y = e.new_ones(pos_edge_index.size(1))
        neg_y = e.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(e, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(e, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)
        recon_loss = F.binary_cross_entropy(input=pred, target=y, reduction='mean')

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        pred_binary = (pred >= 0.5).astype(int)  # Threshold at 0.5 to get binary predictions
        metrics = {
            'recon_loss': float(recon_loss),
            'f1': f1_score(y, pred_binary),
            'auc': roc_auc_score(y, pred),
            'ap': average_precision_score(y, pred)
        }

        return metrics

    def test_nodal_latent(self, z: Tensor, pos_edge_index: Tensor,
             neg_edge_index: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Given latent variables :obj:`z`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes area under the ROC curve (AUC) and average precision (AP)
        scores.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (torch.Tensor): The positive edges to evaluate
                against.
            neg_edge_index (torch.Tensor): The negative edges to evaluate
                against.
        """
        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)
        recon_loss = F.binary_cross_entropy(input=pred, target=y, reduction='mean')

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        pred_binary = (pred >= 0.5).astype(int)  # Threshold at 0.5 to get binary predictions

        metrics = {
            'recon_loss': float(recon_loss),
            'f1': f1_score(y, pred_binary),
            'auc': roc_auc_score(y, pred),
            'ap': average_precision_score(y, pred)
        }

        return metrics

    def test_edge(self, 
                  pos_edge_prob: Tensor, neg_edge_prob: Tensor):
        r"""Given latent variables :obj:`z` which represent the probability of an edge, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes area under the ROC curve (AUC) and average precision (AP)
        scores.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (torch.Tensor): The positive edges to evaluate
                against.
            neg_edge_index (torch.Tensor): The negative edges to evaluate
                against.
        """
        pred = torch.cat([pos_edge_prob, neg_edge_prob], dim=0)
        y = torch.cat([torch.ones(len(pos_edge_prob)), torch.zeros(len(neg_edge_prob))], dim=0)
        recon_loss = F.binary_cross_entropy(pred, y, reduction='mean')
        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        pred_binary = (pred >= 0.5).astype(int)  # Threshold at 0.5 to get binary predictions
        metrics = {
            'recon_loss': float(recon_loss),
            'f1': f1_score(y, pred_binary),
            'auc': roc_auc_score(y, pred),
            'ap': average_precision_score(y, pred)
        }
        return metrics

class VGAE(GAE):
    def __init__(self, encoder: Module, decoder: Module, nodal_latent_dist):
        super().__init__(encoder, decoder)
        self.nodal_latent_dist = nodal_latent_dist
    
    def encode_and_sample(self, x, train_pos_edge_index):
        if self.nodal_latent_dist == 'normal':
            mu, log_stdv = self.encoder(x, train_pos_edge_index)
            distrib = torch.distributions.Normal(mu, torch.exp(log_stdv.clamp(max=MAX_LOGSTD)))
            z = distrib.rsample()
            kl_loss = - 0.5 * torch.mean(torch.sum(1 + 2 * log_stdv - mu**2 - log_stdv.exp()**2, dim=1))
        elif self.nodal_latent_dist == 'ks':
            log_a, log_b = self.encoder(x, train_pos_edge_index)
            distrib = KumaraswamyStable(log_a, log_b)
            z = distrib.rsample()
            kl_loss = - distrib.entropy().sum(dim=1).mean()
        elif self.nodal_latent_dist == 'beta':
            log_alpha, log_beta = self.encoder(x, train_pos_edge_index)
            distrib = torch.distributions.Beta(torch.exp(log_alpha), torch.exp(log_beta))
            z = distrib.rsample()
            kl_loss = - distrib.entropy().sum(dim=1).mean()
        return z, kl_loss
""" 

class VGAE(GAE):
    def __init__(self, encoder: Module, decoder: Optional[Module] = None):
        super().__init__(encoder, decoder)

    def reparametrize(self, mu: Tensor, logstd: Tensor, sample: bool) -> Tensor:
        if sample:
            distrib = torch.distributions.Normal(mu, torch.exp(logstd))
            return distrib.rsample()
            #return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def encode(self, *args, **kwargs) -> Tensor:
        """"""  # noqa: D419
        self.__mu__, self.__logstd__ = self.encoder(*args, **kwargs)
        self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)
        z = self.reparametrize(self.__mu__, self.__logstd__, sample=self.training)
        return z

    def kl_loss(self, 
                mu: Optional[Tensor] = None,
                logstd: Optional[Tensor] = None) -> Tensor:
        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(max=MAX_LOGSTD)
        return -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))
"""

class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True) # cached only for transductive learning
        self.conv_mu = GCNConv(hidden_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(hidden_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
    
class VariationalGCNMLPEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(VariationalGCNMLPEncoder, self).__init__()
        self.shared_conv = GCNConv(in_channels, hidden_channels, cached=True) # cached only for transductive learning
        self.conv1 = GCNConv(hidden_channels, hidden_channels, cached=True) # cached only for transductive learning
        self.conv2 = GCNConv(hidden_channels, hidden_channels, cached=True)
        self.mlp_1 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            #nn.LeakyReLU(),
            #nn.Linear(hidden_channels, hidden_channels),
            nn.LeakyReLU(),
            nn.Linear(hidden_channels, out_channels)
        )
        self.mlp_2 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            #nn.LeakyReLU(),
            #nn.Linear(hidden_channels, hidden_channels),
            nn.LeakyReLU(),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index):
        x = F.leaky_relu(self.shared_conv(x, edge_index))
        param_1 = self.mlp_1(F.leaky_relu(self.conv1(x, edge_index)))
        param_2 = self.mlp_2(F.leaky_relu(self.conv2(x, edge_index)))
        return param_1, param_2
        #return self.mlp_1(self.conv1(x, edge_index).relu()), self.mlp_2(self.conv2(x, edge_index).relu())

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv = GCNConv(in_channels, hidden_channels, cached=True) # cached only for transductive learning
        self.conv_log_a = GCNConv(hidden_channels, out_channels, cached=True)
        self.dropout = nn.Dropout(p=0.5)
        #self.conv_log_b = GCNConv(hidden_channels, out_channels, cached=True)
    def forward(self, x, edge_index):
        x = F.leaky_relu(self.conv(x, edge_index))
        x = self.dropout(x)
        log_a_nodal = F.leaky_relu(self.conv_log_a(x, edge_index))
        #log_b_nodal = F.leaky_relu(self.conv_log_b(x, edge_index))
        return log_a_nodal#, log_b_nodal
    
def posterior_predictive_metrics(post_pred_samples, labels, viz=False):
    #labels = torch.cat([torch.ones(pos_edge_index.shape[1]), torch.zeros(neg_edge_index.shape[1])], dim=0)

    #post_pred_samples = torch.stack(post_pred_samples) # shape (num_samples, num_edges)
    mean, std = post_pred_samples.mean(dim=0), post_pred_samples.std(dim=0)
    error = (mean - labels).abs()
    bs = (mean - labels)**2

    #### metrics
    # quality of the uncertainty
    pearson_corr = torch.corrcoef(torch.stack((error, std)))[0, 1] # shape (1)

    # quality of uncertainty over positive and negative edges
    pos_mask, neg_mask = (labels == 1), (labels == 0)
    #assert len(pos_mask) == len(neg_mask), "Positive and negative masks must be the same length"
    num_unique_error = torch.unique(error).shape[0]
    num_unique_std = torch.unique(std).shape[0]
    print(f"\tUnique error values: {num_unique_error}, unique std values: {num_unique_std}")
    
    pearson_corr_pos = torch.corrcoef(torch.stack((std[pos_mask], error[pos_mask])))[0, 1]
    pearson_corr_neg = torch.corrcoef(torch.stack((std[neg_mask], error[neg_mask])))[0, 1]

    if viz:
        # plot the error vs std for the positive and negative edges
        fig, axs = plt.subplots(2, 1)  # Corrected the order of fig and axs

        # Plotting the scatter plots on respective axes
        axs[0].scatter(std[pos_mask], error[pos_mask], color='blue', label=f'Positive edges {pearson_corr_pos:.2f}')
        axs[1].scatter(std[neg_mask], error[neg_mask], color='red', label=f'Negative edges {pearson_corr_neg:.2f}')

        # Setting labels for each axis
        axs[0].set_ylabel('Error')
        axs[1].set_ylabel('Error')
        axs[1].set_xlabel('Standard deviation')

        # Adding legends to each subplot individually
        axs[0].legend()
        axs[1].legend()

        # Setting the limits for x and y axes
        axs[0].set_xlim(0, 1)
        axs[1].set_xlim(0, 1)
        axs[0].set_ylim(0, 1)
        axs[1].set_ylim(0, 1)

        # Display the plot
        plt.show()

    pred_binary = (mean >= 0.5).int()  # Threshold at 0.5 to get binary predictions
    # quality of mean of posterior predictive as predictor.
    # auc and ap may not make sense, as the mean is not really a probability.
    auc, ap, f1 = roc_auc_score(labels, mean), average_precision_score(labels, mean), f1_score(labels, pred_binary)

    print(f"\tMean predictor performance: AUC: {auc:.4f}, AP: {ap:.4f}, F1: {f1:.4f}")
    print(f"\tPearson correlation: {pearson_corr:.4f}, pos: {pearson_corr_pos:.4f}, neg: {pearson_corr_neg:.4f}")

    metrics = {
        'abs_error': error.mean().float(),
        'brier_score': bs.mean().float(),
        'std': std.mean().float(),
        'pearson_corr': pearson_corr.float(),
        'pearson_corr_pos': pearson_corr_pos.float(),
        'pearson_corr_neg': pearson_corr_neg.float(),
        'auc': auc,
        'ap': ap,
        'f1': f1
    }
    return metrics