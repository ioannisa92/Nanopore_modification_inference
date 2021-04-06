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
conda create -n ndmi_reproduce python=3.6
conda activate ndmi_reproduce
git clone https://github.com/ioannisa92/Nanopore_modification_inference.git
python setup.py install
```
## To run our multi-GPU enabled GridSearch:
We have developed our own multi-GPU enables grid search with cross validation. \
Each parameter combination can be run on a single gpu, thereby accelerating the search for the architecture. 

The following line of code accepts a kmer model (`-i`) table and the number of cross validation folds to be run (`-k`). \
The optimal parameter set is determined by the parameter set that achieves the best average RMSE across all folds. 
```
python gscv_main.py -i ./ont_models/r9.4_180mv_70bps_5mer_5to3_RNA.model -k 10
python gscv_main.py -i ./ont_models/r9.4_180mv_450bps_6mer_DNA.model -k 10
```

## Downsample Analysis
We have conducted four different kinds of downsample (dropping information from the training data) \
analysis to test the limits of our model's performance:
1. Random kmer downsampling in a 50-fold cross-validation fashion.
2. Base dropout, where each base is dropped regardless of position on the kmer.
3. Position-base dropout, where each base is dropped from each of the kmer's positions. 
4. Combination dropout, where a pair of bases is dropped.

In each case, the dropped data appear in the test set.

The `cv_main.py` script runs 1 & 3:
```
usage: cv_main.py [-h] [-i FILE] [-cv] [-k FOLDS] [-o OUT] [-v VERBOSITY]
                  [-kmer_cv] [-test_splits SPLITS [SPLITS ...]]

Script takes in a kmer and pA measurement file. The user can select between
random cross validation, or targeted cross validation, where each based is
hidden from each position of the kmer in training. Script saves cross
validation results as a .npy file

optional arguments:
  -h, --help            show this help message and exit
  -i FILE, --FILE FILE  kmer file with pA measurement
  -cv, --CV             MODE: Random CV splits of variable size
  -k FOLDS, --FOLDS FOLDS
                        K for fold numbers in cross validation
  -o OUT, --OUT OUT     Full path for .npy file where results are saved
  -v VERBOSITY, --VERBOSITY VERBOSITY
                        Verbosity of model. Other than zero, loss per batch
                        per epoch is printed. Default is 0, meaning nothing is
                        printed
  -kmer_cv, --KMERCV    MODE: Position-based dropout of each base
  -test_splits SPLITS [SPLITS ...], --SPLITS SPLITS [SPLITS ...]
                        Test splits to run k-fold cross validation over
```

For example to run the kmer downsample analysis on DNA:
```
python cv_main.py -i ./ont_models/r9.4_180mv_450bps_6mer_DNA.model -cv -o dna_downsample_results.npy
```
and the positional dropout analysis:
```
python cv_main.py -i ./ont_models/r9.4_180mv_450bps_6mer_DNA.model -kmer_cv -o dna_posdrop_results.npy
```

The `exclude_bases.py` script runs 2 & 4:
```
usage: exclude_bases.py [-h] [-i FILE] [-base_pair_exclude] [-base_exclude]
                        [-o OUT] -n_type NTYPE

Run base-pair specific dropout cross validation

optional arguments:
  -h, --help            show this help message and exit
  -i FILE, --FILE FILE  kmer file with pA measurement
  -base_pair_exclude, --PAIRS
                        MODE: pairs of pases will be excluded
  -base_exclude, --SOLO
                        MODE: each of the four bases will be removed from
                        training
  -o OUT, --OUT OUT     Full path for .npy file where results are saved
  -n_type NTYPE, --NTYPE NTYPE
                        Type of nucleotide examined: DNA or RNA
```

For example to run the base dropout analysis on DNA:
```
python exclude_bases.py -i ./ont_models/r9.4_180mv_450bps_6mer_DNA.model -o dna_exclude_base_results.npy -n_type $3 'DNA' -base_exclude
```
and the base-pair dropout analysis:
```
python exclude_bases.py -i ./ont_models/r9.4_180mv_450bps_6mer_DNA.model -o dna_exclude_basepairs_results.npy -n_type $3 'DNA' -base_pair_exclude
```

## Modification prediction analysis
To train the model on only canonical kmers, and predict on all possible M (methylated C) modified kmers run the following line:
```
python dna_mod_pred.py -i  ./ont_models/r9.4_180mv_450bps_6mer_DNA.model -model_fn dna_model -o dna_mod_pred_50repeat_results.npy
```

To train the model on canonical and fractions of M containing kmers, and then predict all possible M modified kmers run the following line:
```
python dna_mod_trainpred.py -i ./ont_models/r9.4_180mv_450bps_6mer_DNA.model  -o dna_mod_trainpred_results_50fold.npy
```

## Reproducing paper results
The manuscript's results can be reproduced at once by simply running the following code:
```
bash run_all_analysis.sh
```
