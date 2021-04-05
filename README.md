# Nanopore modification inference
Pre-print: [Towards Inferring Nanopore Sequencing Ionic Currents from Nucleotide Chemical Structures](https://www.biorxiv.org/content/10.1101/2020.11.30.404947v1.abstract)


## Introduction
We develop a model that associates chemical information on nanopore kmer models with their mean pA value. We showcase that the model can 
learn chemical information in specific contexts which can be transferred to identify _de_novo_ kmer modifications. Our model is implemented with keras.

## Installation
Our model runs with python3. We recommend to use a recent version of python3 (eg. python>=3.6). \
We recommend using conda to create a virtual environment. \
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


## Reproducing paper results
The manuscript's results can be reproduced by simply running the following code:
'''
bash run_all_analysis
'''
