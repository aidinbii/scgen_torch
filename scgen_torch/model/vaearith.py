import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy
from scipy import sparse

from .modules import Encoder, Decoder
from .util import balancer, extractor

class vaeArith(nn.Module):
    """VAE with Arithmetic vector Network class. This class contains the implementation of Variational Auto-encoder network with Vector Arithmetics.
       Parameters
       ----------
       input_dim: Integer
            Number of input features (i.e. gene in case of scRNA-seq).
       hidden_layer_sizes: List
            A list of hidden layer sizes for encoder network. Decoder network will be the reversed order.
       latent_dim: Integer
            Bottleneck layer (z)  size.
       dr_rate: Float
            Dropout rate applied to all layers, if `dr_rate`==0 no dropout will be applied.
       use_bn: Boolean
            If `True` batch normalization will be applied to layers.
       use_ln: Boolean
            If `True` layer normalization will be applied to layers.
    """

    def __init__(self, input_dim: int, hidden_layer_sizes: list = [800, 800], z_dimension = 100, dr_rate: float = 0.2, use_cuda = True, **kwargs):
        super().__init__()
        assert isinstance(hidden_layer_sizes, list)
        assert isinstance(z_dimension, int)

        self.x_dim = input_dim
        self.z_dim = z_dimension
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dr_rate = dr_rate

        self.encoder = Encoder(self.x_dim, self.hidden_layer_sizes, self.z_dim, self.dr_rate)
        self.decoder = Decoder(self.z_dim, self.hidden_layer_sizes, self.x_dim, self.dr_rate)


        self.alpha = kwargs.get("alpha", 0.00005)

    def _sample_z(self, mu: torch.Tensor, log_var: torch.Tensor):
        """
            Samples from standard Normal distribution with shape [size, z_dim] and
            applies re-parametrization trick. It is actually sampling from latent
            space distributions with N(mu, var) computed in `_encoder` function.

            # Parameters
                mean and log_var.


            # Returns
                The computed Tensor of samples with shape [size, z_dim].

        Reparameterization trick to sample from N(mu, var) from
            N(0,1).
            :param mu: (Tensor) Mean of the latent Gaussian [B x D]
            :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
            :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def to_latent(self, data: torch.Tensor) -> torch.Tensor:
        """
            Map `data` in to the latent space. This function will feed data
            in encoder part of VAE and compute the latent space coordinates
            for each sample in data.

            # Parameters
                data:  numpy nd-array
                    Numpy nd-array to be mapped to latent space. `data.X` has to be in shape [n_obs, n_vars].

            # Returns
                latent: numpy nd-array
                    Returns array containing latent space encoding of 'data'
        """
        if not torch.is_tensor(data):
            data = torch.tensor(data) # to tensor
        mu, logvar = self.encoder(data)
        latent = self._sample_z(mu, logvar)
        return latent

    def _avg_vector(self, data: torch.Tensor) -> torch.Tensor:
        """
            Computes the average of points which computed from mapping `data`
            to encoder part of VAE.

            # Parameters
                data:  numpy nd-array
                    Numpy nd-array matrix to be mapped to latent space. Note that `data.X` has to be in shape [n_obs, n_vars].

            # Returns
                The average of latent space mapping in numpy nd-array.

        """
        #data = torch.tensor(data) # to tensor
        latent = self.to_latent(data)
        latent_avg = torch.mean(latent, dim=0) # maybe keepdim = True, so that shape (,1)
        return latent_avg

    def reconstruct(self, data, use_data=False):
        """
            Map back the latent space encoding via the decoder.

            # Parameters
                data: `~anndata.AnnData`
                    Annotated data matrix whether in latent space or gene expression space.
                use_data: bool
                    This flag determines whether the `data` is already in latent space or not.
                    if `True`: The `data` is in latent space (`data.X` is in shape [n_obs, z_dim]).
                    if `False`: The `data` is not in latent space (`data.X` is in shape [n_obs, n_vars]).

            # Returns
                rec_data: 'numpy nd-array'
                    Returns 'numpy nd-array` containing reconstructed 'data' in shape [n_obs, n_vars].
        """
        if not torch.is_tensor(data):
            data = torch.tensor(data) # to tensor

        if use_data:
            rec_data = self.decoder(data)
        else:
            latent = self.to_latent(data)
            rec_data = self.decoder(latent)
        return rec_data

    def _loss_function(self, x: torch.Tensor, xhat: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor):
        """
            Defines the loss function of VAE network after constructing the whole
            network. This will define the KL Divergence and Reconstruction loss for
            VAE and also defines the Optimization algorithm for network. The VAE Loss
            will be weighted sum of reconstruction loss and KL Divergence loss.

            # Parameters
                No parameters are needed.

            # Returns
                Nothing will be returned.
        """
        kl_loss = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2)) # check dimensions
        recons_loss = F.mse_loss(xhat, x)
        vae_loss = recons_loss + self.alpha * kl_loss
        return vae_loss


    def forward(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        z = self._sample_z(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar

    def predict(self, adata, conditions, cell_type_key, condition_key, adata_to_predict=None, celltype_to_predict=None,
              obs_key="all", biased=False):
        """
            Predicts the cell type provided by the user in stimulated condition.

            # Parameters
                celltype_to_predict: basestring
                    The cell type you want to be predicted.
                obs_key: basestring or dict
                    Dictionary of celltypes you want to be observed for prediction.
                adata_to_predict: `~anndata.AnnData`
                    Adata for unpertubed cells you want to be predicted.

            # Returns
                predicted_cells: numpy nd-array
                    `numpy nd-array` of predicted cells in primary space.
                delta: float
                    Difference between stimulated and control cells in latent space

            # Example
            ```python
                import anndata
                import scgen
                train_data = anndata.read("./data/train.h5ad"
                validation_data = anndata.read("./data/validation.h5ad")
                network = scgen.VAEArith(x_dimension= train_data.shape[1], model_path="./models/test" )
                network.train(train_data=train_data, use_validation=True, valid_data=validation_data, shuffle=True, n_epochs=2)
        pred, delta = scg.predict(adata= train_new,conditions={"ctrl": "control", "stim":"stimulated"},
                          cell_type_key="cell_type",condition_key="condition",adata_to_predict=unperturbed_cd4t)            ```
        """
        if obs_key == "all":
            ctrl_x = adata[adata.obs[condition_key] == conditions["ctrl"], :]
            stim_x = adata[adata.obs[condition_key] == conditions["stim"], :]
            if not biased:
                ctrl_x = balancer(ctrl_x, cell_type_key=cell_type_key, condition_key=condition_key)
                stim_x = balancer(stim_x, cell_type_key=cell_type_key, condition_key=condition_key)
        else:
            key = list(obs_key.keys())[0]
            values = obs_key[key]
            subset = adata[adata.obs[key].isin(values)]
            ctrl_x = subset[subset.obs[condition_key] == conditions["ctrl"], :]
            stim_x = subset[subset.obs[condition_key] == conditions["stim"], :]
            if len(values) > 1 and not biased:
                ctrl_x = balancer(ctrl_x, cell_type_key=cell_type_key, condition_key=condition_key)
                stim_x = balancer(stim_x, cell_type_key=cell_type_key, condition_key=condition_key)
        if celltype_to_predict is not None and adata_to_predict is not None:
            raise Exception("Please provide either a cell type or adata not both!")
        if celltype_to_predict is None and adata_to_predict is None:
            raise Exception("Please provide a cell type name or adata for your unperturbed cells")
        if celltype_to_predict is not None:
            ctrl_pred = extractor(adata, celltype_to_predict, conditions, cell_type_key, condition_key)[1]
        else:
            ctrl_pred = adata_to_predict
        if not biased:
            eq = min(ctrl_x.X.shape[0], stim_x.X.shape[0])
            cd_ind = numpy.random.choice(range(ctrl_x.shape[0]), size=eq, replace=False)
            stim_ind = numpy.random.choice(range(stim_x.shape[0]), size=eq, replace=False)
        else:
            cd_ind = numpy.random.choice(range(ctrl_x.shape[0]), size=ctrl_x.shape[0], replace=False)
            stim_ind = numpy.random.choice(range(stim_x.shape[0]), size=stim_x.shape[0], replace=False)
        if sparse.issparse(ctrl_x.X) and sparse.issparse(stim_x.X):
            latent_ctrl = self._avg_vector(ctrl_x.X.A[cd_ind, :])
            latent_sim = self._avg_vector(stim_x.X.A[stim_ind, :])
        else:
            latent_ctrl = self._avg_vector(ctrl_x.X[cd_ind, :])
            latent_sim = self._avg_vector(stim_x.X[stim_ind, :])
            delta = latent_sim - latent_ctrl
        if sparse.issparse(ctrl_pred.X):
            latent_cd = self.to_latent(ctrl_pred.X.A)
        else:
            latent_cd = self.to_latent(ctrl_pred.X)
        stim_pred = delta + latent_cd
        predicted_cells = self.reconstruct(stim_pred, use_data=True)
        return predicted_cells, delta


    def restore_model(self):
        """
            restores model weights from `model_to_use`.

            # Parameters
                No parameters are needed.

            # Returns
                Nothing will be returned.

            # Example
            ```python
                import anndata
                import scgen
                train_data = anndata.read("./data/train.h5ad")
                validation_data = anndata.read("./data/validation.h5ad")
                network = scgen.VAEArith(x_dimension= train_data.shape[1], model_path="./models/test" )
                network.restore_model()
            ```
        """
        checkpoint = torch.load(self.model_to_use)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        #+END_SRC
