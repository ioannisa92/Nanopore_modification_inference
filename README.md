# Nanopore modification inference
### [Towards Inferring Nanopore Sequencing Ionic Currents from Nucleotide Chemical Structures](https://www.biorxiv.org/content/10.1101/2020.11.30.404947v1.abstract)


## Introduction
We develop a model that associates chemical information on nanopore kmer models with their mean pA value. We showcase that the model can 
chemical information in specific contexs which can be transferred to identify _de_novo_ kmer modifications. Our model is implemented with keras.

## Installation
Our model runs with python3. We recommend to use a recent version of python3 (eg. python>=3.6). 
We recommend using conda to create a virtual environment.
Follow the following steps to install and replicate our results:

```bash
git clone https://github.com/ioannisa92/Nanopore_modification_inference.git
```
Then run in the VEGA repository:
```bash
python setup.py install
```

_Note: We recommend to make a new environment to run VEGA, eg. using Anaconda:_
```bash
conda create -n vega-test python=3.7.0
```

## Getting started
VEGA needs 2 things to analyze your data:


* A single-cell dataset wrapped using the Scanpy package (Wolf et al. 2018)
* A GMT file specifying the gene module variables (GMVs) and gene membership, eg. from MSigDB

We recommend that the Scanpy Anndata object is preprocessed before passed to VEGA:
```python
import scanpy as sc
adata = sc.read(path)
sc.pp.normalize_per_cell(adata)
sc.pp.log1p(adata)
```
We also recommend using a subset of highly variable genes (5000-7000). See the [Scanpy documentation](https://scanpy.readthedocs.io/en/stable/index.html) for more information on preprocessing.


## Tutorial
A tutorial is available [here](https://github.com/LucasESBS/vega/blob/main/tutorials/Vega-tutorial.ipynb). It will guide the user through the different steps of the analysis with VEGA.

## Reproducing paper results
VEGA manuscript results can be reproduced using [the following code](https://github.com/LucasESBS/vega-reproducibility).

## Manuscript
VEGA preprint can be found [here](https://www.biorxiv.org/content/10.1101/2020.12.17.423310v1.abstract).
