import torch
import torch.nn as nn
from torch.nn import functional as F


class Encoder(nn.Module):
    """
    Constructs the encoder sub-network of VAE. This class implements the
    encoder part of Variational Auto-encoder. It will transform primary
    data in the `n_vars` dimension-space to means and log variances of `z_dimension` latent space.

    Parameters
    ----------
    x_dimension: integer
        number of gene expression space dimensions.
    layer_sizes: List
        List of hidden layer sizes.
    z_dimension: integer
        number of latent space dimensions.
    dropout_rate: float
        dropout rate
    """

    def __init__(self, x_dimension: int, layer_sizes: list, z_dimension: int, dropout_rate: float):
        super().__init__() # to run nn.Module's init method

        self.x_dim = x_dimension
        self.z_dim = z_dimension

        # encoder architecture
        self.linear1 = nn.Linear(in_features = self.x_dim, out_features = layer_sizes[0], bias = False) # why without bias? which xavier weight init. to use (uniform or normal)
        self.bn1 = nn.BatchNorm1d(num_features = layer_sizes[0])
        self.dropout1 = nn.Dropout(p = dropout_rate)
        self.linear2 = nn.Linear(layer_sizes[0], layer_sizes[1], bias = False)
        self.bn2 = nn.BatchNorm1d(num_features = layer_sizes[1])
        self.dropout2 = nn.Dropout(p = dropout_rate)
        self.linear3_mean = nn.Linear(layer_sizes[1], self.z_dim)
        self.linear3_var = nn.Linear(layer_sizes[1], self.z_dim)
        # or one layer; what is the differenc?
        # self.linear3 = nn.Linear(800, 2 * self.z_dim) # e.g. 100 for means and 100 for var. Possible answer: two layers -> different sets of weights for mean and var!

    def forward(self, x: torch.Tensor):
        x = F.leaky_relu(self.bn1(self.linear1(x)))  # check the default values for parameters in TF and pytorch, e.g. leaky_relu. They could be different!
        x = self.dropout1(x)
        x = F.leaky_relu(self.bn2(self.linear2(x)))
        x = self.dropout2(x)
        mean = self.linear3_mean(x) # do I need to reshape (x.view()) smth?
        log_var = self.linear3_var(x)
        return mean, log_var


class Decoder(nn.Module):
    """
            Constructs the decoder sub-network of VAE. This class implements the
            decoder part of Variational Auto-encoder. Decodes data from latent space to data space. It will transform constructed latent space to the previous space of data with n_dimensions = n_vars.

            # Parameters
               z_dimension: integer
               number of latent space dimensions.
               layer_sizes: List
               List of hidden layer sizes.
               x_dimension: integer
               number of gene expression space dimensions.
               dropout_rate: float
               dropout rate


        """
    def __init__(self, z_dimension: int, layer_sizes: list, x_dimension: int, dropout_rate: float):
        super().__init__()

        self.z_dim = z_dimension
        self.x_dim = x_dimension

        # decoder architecture
        self.linear4 = nn.Linear(in_features = self.z_dim, out_features = layer_sizes[0], bias = False) # why without bias? which xavier weight init. to use (uniform or normal)
        self.bn3 = nn.BatchNorm1d(num_features = layer_sizes[0])
        self.dropout3 = nn.Dropout(p = dropout_rate)
        self.linear5 = nn.Linear(layer_sizes[0], layer_sizes[1], bias = False)
        self.bn4 = nn.BatchNorm1d(num_features = layer_sizes[1])
        self.dropout4 = nn.Dropout(p = dropout_rate)
        self.linear6 = nn.Linear(layer_sizes[1], self.x_dim)


    def forward(self, x: torch.Tensor):
        x = F.leaky_relu(self.bn3(self.linear4(x)))
        # check the default values for parameters in TF and pytorch, e.g. leaky_relu.
        # They could be different!
        x = self.dropout3(x)
        x = F.leaky_relu(self.bn4(self.linear5(x)))
        x = self.dropout4(x)
        x = F.relu(self.linear6(x)) # do I need to reshape (x.view()) smth?
        return x
