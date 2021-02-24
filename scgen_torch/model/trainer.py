import torch
from scipy import sparse
from anndata import AnnData

from .util import shuffle_adata, balancer, extractor

import logging
log = logging.getLogger(__name__)


class vaeArithTrainer:
    """
    Trains the network `n_epochs` times with given `train_data`
    and validates the model using validation_data if it was given
    in the constructor function. This function is using `early stopping`
    technique to prevent over-fitting.

    # Parameters
                train_data: scanpy AnnData
                    Annotated Data Matrix for training VAE network.
                use_validation: bool
                    if `True`: must feed a valid AnnData object to `valid_data` argument.
                valid_data: scanpy AnnData
                    Annotated Data Matrix for validating VAE network after each epoch.
                n_epochs: int
                    Number of epochs to iterate and optimize network weights
                batch_size: integer
                    size of each batch of training dataset to be fed to network while training.
                early_stop_limit: int
                    Number of consecutive epochs in which network loss is not going lower.
                    After this limit, the network will stop training.
                threshold: float
                    Threshold for difference between consecutive validation loss values
                    if the difference is upper than this `threshold`, this epoch will not
                    considered as an epoch in early stopping.
                initial_run: bool
                    if `True`: The network will initiate training and log some useful initial messages.
                    if `False`: Network will resume the training using `restore_model` function in order
                        to restore last model which has been trained with some training dataset.
                shuffle: bool
                    if `True`: shuffles the training dataset

    # Returns
                Nothing will be returned

    # Example
    ```python
                import anndata
                import scgen
                train_data = anndata.read("./data/train.h5ad"
                validation_data = anndata.read("./data/validation.h5ad"
                network = scgen.VAEArith(x_dimension= train_data.shape[1], model_path="./models/test")
                network.train(train_data=train_data, use_validation=True, valid_data=validation_data, shuffle=True, n_epochs=2)
    ```
    """

    def __init__(self, model, adata, n_epochs, batch_size, save, verbose, use_validation=False, early_stop_limit = 20, threshold=0.0025, initial_run=True, shuffle=True,  **kwargs): # maybe add more parameters

        # super().__init__()

        self.model = model

        #self.use_cuda = use_cuda and torch.cuda.is_available()
        #if self.use_cuda:
        #   self.model.cuda()

        self.seed = kwargs.get("seed", 2021)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            self.model.cuda()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



        self.adata = adata

        self.shuffle = shuffle
        self.batch_size = batch_size
        self.initial_run = initial_run
        self.early_stop_limit = early_stop_limit
        self.use_validation = use_validation
        self.threshold = threshold
        self.save = save
        self.verbose = verbose

        # Optimization attributes
        self.optim = None
        # self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        # self.epoch = -1  # epoch = self.epoch + 1 in compute metrics
        # self.training_time = 0
        # self.n_iter = 0

        self.model_to_use = kwargs.get("model_path", "example_model_save_2.pth")

    @staticmethod
    def _anndataToTensor(adata: AnnData) -> torch.Tensor:
        data_ndarray = adata.X.A
        data_tensor = torch.from_numpy(data_ndarray)
        return data_tensor

    @staticmethod
    def train_valid_split(adata: AnnData, proportion_train = 0.75):
        n_obs = adata.shape[0]
        shuffled = shuffle_adata(adata)

        train_adata = shuffled[:int(proportion_train * n_obs)] # maybe not the best way to round
        valid_adata = shuffled[int(proportion_train * n_obs):]
        return train_adata, valid_adata


    def train(self, lr=1e-3, eps=0.01, params=None, use_validation = False, **extras_kwargs):
        if self.initial_run:
            log.info("----Training----")
        if not self.initial_run:
            self.restore_model()

        if params is None:
            params = filter(lambda p: p.requires_grad, self.model.parameters())

        self.optim = torch.optim.Adam(
            params, lr=lr) # consider changing the param. like weight_decay, eps, etc.

        train_data, valid_data = self.train_valid_split(self.adata) # possible bad of using static method this way. Think about adding static methods to util.py

        if self.shuffle:
            train_adata = shuffle_adata(train_data)
            valid_adata = shuffle_adata(valid_data)
            loss_hist = []
            patience = self.early_stop_limit
            min_delta = self.threshold
            patience_cnt = 0
        for epoch in range(self.n_epochs):
            #super(VAEArith, self.model).train() # put model to training mode and .super() to use method train() from nn.Module class
            self.model.train()
            train_loss = 0
            loss_hist.append(0)
            for lower in range(0, train_adata.shape[0], self.batch_size):
                upper = min(lower + self.batch_size, train_adata.shape[0])
                if sparse.issparse(train_adata.X):
                    x_mb = torch.from_numpy(train_adata[lower:upper, :].X.A)
                else:
                    x_mb = torch.from_numpy(train_adata[lower:upper, :].X)
                if upper - lower > 1:
                    x_mb = x_mb.to(self.device)
                    reconstructions, mu, logvar = self.model(x_mb)
                    loss = self.model._loss_function(x_mb, reconstructions, mu, logvar)

                    self.optim.zero_grad()

                    loss.backward()
                    self.optim.step()

                    train_loss += loss.item() # loss.item() contains the loss of entire mini-batch divided by the batch size


            if use_validation:
                #super(VAEArith, self.model).train(False) # changes the behavior of some layers (e.g. dropout will be disabled and the running stats of batchnorm layers will be used); will notify all your layers that you are in eval mode, that way, batchnorm or dropout layers will work in eval mode instead of training mode.
                self.model.eval()
                valid_loss = 0
                train_loss_end_epoch = 0
                with torch.no_grad(): # disables the gradient calculation
                    for lower in range(0, train_adata.shape[0], self.batch_size):
                        upper = min(lower + self.batch_size, train_adata.shape[0])
                        if sparse.issparse(train_adata.X):
                            x_mb = torch.from_numpy(train_adata[lower:upper, :].X.A)
                        else:
                            x_mb = torch.from_numpy(train_adata[lower:upper, :].X)
                        if upper - lower > 1:
                            x_mb = x_mb.to(self.device)
                            reconstructions, mu, logvar = self.model(x_mb)
                            loss = self.model._loss_function(x_mb, reconstructions, mu, logvar)

                            train_loss_end_epoch += loss.item() # loss.item() contains the loss of entire mini-batch divided by the batch size

                    for lower in range(0, valid_adata.shape[0], self.batch_size):
                        upper = min(lower + self.batch_size, valid_adata.shape[0])
                        if sparse.issparse(valid_adata.X):
                            x_mb = torch.from_numpy(valid_adata[lower:upper, :].X.A)
                        else:
                            x_mb = torch.from_numpy(valid_adata[lower:upper, :].X)
                        if upper - lower > 1:
                            x_mb = x_mb.to(self.device)
                            reconstructions, mu, logvar = self.model(x_mb)
                            loss = self.model._loss_function(x_mb, reconstructions, mu, logvar)

                            valid_loss += loss.item() # loss.item() contains the loss of entire mini-batch divided by the batch size
                if self.verbose:
                    print(f"Epoch: {epoch}. Train Loss: {train_loss_end_epoch / (1)} Validation Loss: {valid_loss / (1)}")
            else:
                if self.verbose:
                    print(f"Epoch: {epoch}. Train Loss: {train_loss / (train_adata.shape[0] )}")
        if self.save:
            #os.makedirs(self.model_to_use, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optim.state_dict(),
                'loss': train_loss}, self.model_to_use)
            log.info(f"Model saved in file: {self.model_to_use}. Training finished")
#+END_SRC
