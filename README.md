# Nanopore modification inference
Pre-print: [Towards Inferring Nanopore Sequencing Ionic Currents from Nucleotide Chemical Structures](https://www.biorxiv.org/content/10.1101/2020.11.30.404947v1.abstract)


## Introduction
We develop a model that associates chemical information on nanopore kmer models with their mean pA value. We showcase that the model can 
learn chemical information in specific contexts which can be transferred to identify _de_novo_ kmer modifications. Our model is implemented with keras.

## Installation
Our model runs with python3. We recommend to use a recent version of python3 (eg. python>=3.6). \
We recommend using conda to create a virtual environment. \
Follow the steps bellow to install and replicate our results:

```
conda create -n ndmi_reproduce python=3.6.0
git clone https://github.com/ioannisa92/Nanopore_modification_inference.git
python setup.py install
```
## To run our multi-GPU enabled GridSearch:
We have developed our own multi-GPU enables grid search with cross validation. \
Each parameter combination can be run on a single gpu, thereby accelerating the search for the architecture. 

The following line of code accepts a kmer model (`-i`) table and the number of cross validation folds to be run (`-k`). \
The optimal parameter set is determined by the parameter set that achieves the best average RMSE across all folds. 
```
python gscv_main.py -i ./ont_models/r9.4_180mv_450bps_6mer_DNA.model -k 10
```

## Reproducing paper results
The manuscript's results can be reproduced by simply running the following code:
```
bash run_all_analysis.sh
```
