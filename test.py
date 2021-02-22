import scanpy as sc
import torch
import scgen_torch as scg

## data
#train_data = read_load.load_file("./data/train_data.h5ad") # Read and load the training dataset (AnnData object)

train_data = sc.read("./scgen_torch/data/train_data.h5ad") # Read and load the training dataset (AnnData object)


## create vaeArith model and train it on the example dataset
vaearith = scg.model.VAEARITH(adata = train_data)

vaearith.train()
