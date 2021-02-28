import inspect
import os
import torch
import pickle
import numpy as np

from anndata import AnnData, read
from typing import Optional, Union

from .vaearith import vaeArith
from .trainer import vaeArithTrainer
from ._utils import _validate_var_names

class BaseMixin:
    """ Adapted from
        Title: scvi-tools
        Authors: Romain Lopez <romain_lopez@gmail.com>,
                 Adam Gayoso <adamgayoso@berkeley.edu>,
                 Galen Xing <gx2113@columbia.edu>
        Date: 14.12.2020
        Code version: 0.8.0-beta.0
        Availability: https://github.com/YosefLab/scvi-tools
        Link to the used code:
        https://github.com/YosefLab/scvi-tools/blob/0.8.0-beta.0/scvi/core/models/base.py
    """
    def _get_user_attributes(self):
        # returns all the self attributes defined in a model class, eg, self.is_trained_
        attributes = inspect.getmembers(self, lambda a: not (inspect.isroutine(a)))
        attributes = [
            a for a in attributes if not (a[0].startswith("__") and a[0].endswith("__"))
        ]
        attributes = [a for a in attributes if not a[0].startswith("_abc_")]
        return attributes

    def _get_public_attributes(self):
        public_attributes = self._get_user_attributes()
        public_attributes = {a[0]: a[1] for a in public_attributes if a[0][-1] == "_"}
        return public_attributes

    def save(
        self,
        dir_path: str,
        overwrite: bool = False,
        save_anndata: bool = False,
        **anndata_write_kwargs,
    ):
        """Save the state of the model.
           Neither the trainer optimizer state nor the trainer history are saved.
           Parameters
           ----------
           dir_path
                Path to a directory.
           overwrite
                Overwrite existing data or not. If `False` and directory
                already exists at `dir_path`, error will be raised.
           save_anndata
                If True, also saves the anndata
           anndata_write_kwargs
                Kwargs for anndata write function
        """
        # get all the public attributes
        public_attributes = self._get_public_attributes()
        # save the model state dict and the trainer state dict only
        if not os.path.exists(dir_path) or overwrite:
            os.makedirs(dir_path, exist_ok=overwrite)
        else:
            raise ValueError(
                "{} already exists. Please provide an unexisting directory for saving.".format(
                    dir_path
                )
            )

        if save_anndata:
            self.adata.write(
                os.path.join(dir_path, "adata.h5ad"), **anndata_write_kwargs
            )

        model_save_path = os.path.join(dir_path, "model_params.pt")
        attr_save_path = os.path.join(dir_path, "attr.pkl")
        varnames_save_path = os.path.join(dir_path, "var_names.csv")

        var_names = self.adata.var_names.astype(str)
        var_names = var_names.to_numpy()
        np.savetxt(varnames_save_path, var_names, fmt="%s")

        torch.save(self.model.state_dict(), model_save_path)
        with open(attr_save_path, "wb") as f:
            pickle.dump(public_attributes, f)

    def _load_expand_params_from_dict(self, state_dict):
        load_state_dict = state_dict.copy()

        device = next(self.model.parameters()).device

        new_state_dict = self.model.state_dict()
        for key, load_ten in load_state_dict.items():
            new_ten = new_state_dict[key]
            if new_ten.size() == load_ten.size():
                continue
            # new categoricals changed size
            else:
                load_ten = load_ten.to(device)
                dim_diff = new_ten.size()[-1] - load_ten.size()[-1]
                fixed_ten = torch.cat([load_ten, new_ten[..., -dim_diff:]], dim=-1)
                load_state_dict[key] = fixed_ten

        self.model.load_state_dict(load_state_dict)

    @classmethod
    def _load_params(
        cls,
        dir_path: str,
        map_location: Optional[str] = None
    ):
        setup_dict_path = os.path.join(dir_path, "attr.pkl")
        model_path = os.path.join(dir_path, "model_params.pt")
        varnames_path = os.path.join(dir_path, "var_names.csv")

        with open(setup_dict_path, "rb") as handle:
            attr_dict = pickle.load(handle)

        model_state_dict = torch.load(model_path, map_location=map_location)

        var_names = np.genfromtxt(varnames_path, delimiter=",", dtype=str)

        return attr_dict, model_state_dict, var_names

    @classmethod
    def load(
        cls,
        dir_path: str,
        adata: Optional[AnnData] = None,
    ):
        """Instantiate a model from the saved output.
           Parameters
           ----------
           dir_path
                Path to saved outputs.
           adata
                AnnData object.
                If None, will check for and load anndata saved with the model.
           Returns
           -------
                Model with loaded state dictionaries.
        """
        adata_path = os.path.join(dir_path, "adata.h5ad")

        load_adata = adata is None

        if os.path.exists(adata_path) and load_adata:
            adata = read(adata_path)
        elif not os.path.exists(adata_path) and load_adata:
            raise ValueError("Save path contains no saved anndata and no adata was passed.")

        attr_dict, model_state_dict, var_names = cls._load_params(dir_path)

        _validate_var_names(adata, var_names)
        cls._validate_adata(adata, attr_dict)
        init_params = cls._get_init_params_from_dict(attr_dict)

        model = cls(adata, **init_params)
        model.model.load_state_dict(model_state_dict)
        model.model.eval()

        model.is_trained_ = attr_dict['is_trained_']

        return model


class VAEARITH(BaseMixin):
    """VAE with Arithmetic vector Network class. This class contains the implementation of Variational Auto-encoder network with Vector Arithmetics.
       Parameters
       ----------
        adata: : `~anndata.AnnData`
            Annotated data matrix. normalized log transformed data for 'mse' loss.
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

    def __init__(self, adata: AnnData, hidden_layer_sizes: list = [800, 800], z_dimension = 100, dr_rate: float = 0.2):
        self.adata = adata

        self.x_dim = adata.n_vars

        self.z_dim = z_dimension
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dr_rate = dr_rate

        self.model = vaeArith(self.x_dim, self.hidden_layer_sizes, self.z_dim, self.dr_rate)

        #self.alpha = kwargs.get("alpha", 0.00005)

        self.is_trained_ = False
        self.trainer = None

    def train(self, n_epochs = 25, batch_size = 32, save = True, verbose = True):
        self.trainer = vaeArithTrainer(self.model, self.adata, n_epochs, batch_size, save, verbose)
        self.trainer.train()
        self.is_trained_ = True


    def to_latent(self, data):
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
        device = next(self.model.parameters()).device # to cpu
        data = torch.tensor(data, device=device) # to tensor
        latent = self.model.to_latent(data)
        latent = latent.cpu().detach()
        return np.array(latent)


    def reconstruct(self, data, use_data):
        device = next(self.model.parameters()).device # to cpu
        data = torch.tensor(data, device=device) # to tensor
        rec_data = self.model.reconstruct(data, use_data)
        rec_data = rec_data.cpu().detach()
        return np.array(rec_data)


    def predict(self, adata, conditions, cell_type_key, condition_key, adata_to_predict=None, celltype_to_predict=None, obs_key="all"):
        device = next(self.model.parameters()).device # to cpu
        #adata_tensor = torch.tensor(adata, device=device) # to tensor
        output = self.model.predict(adata, conditions, cell_type_key, condition_key, adata_to_predict, celltype_to_predict, obs_key)
        prediction = output[0].cpu().detach()
        delta = output[1].cpu().detach()
        return np.array(prediction), np.array(delta)
