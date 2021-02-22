# scGen

## Introduction
A PyTorch implementation of scGen. scGen is a generative model to predict single-cell perturbation response across cell types, studies and species
  [(Nature Methods, 2019)](https://www.nature.com/articles/s41592-019-0494-8) .

## Try
Create VAEArith model and train it on the example dataset.
```
import scanpy as sc
import torch
import scgen_torch as scg

## data
train_data = sc.read("./scgen_torch/data/train_data.h5ad") # Read and load the training dataset (AnnData object)

## create vaeArith model and train it
vaearith = scg.model.VAEARITH(adata = train_data)

vaearith.train()
```
