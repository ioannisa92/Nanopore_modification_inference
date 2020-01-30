#! /usr/bin/env python

import argparse
from modules import kmer_chemistry
from modules.nn_model import *
from modules.kmer_cv import *
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import tqdm
import os
#imports

def rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_pred - y_true)))

def get_AX(kmer_list):

    '''
    Function takes in a kmer-pA measurement lists. Kmers are converted to their SMILES representation.
    An array of the molecular Adjacency and Feature matrices is returned

    Parameters
    -----------
    kmer_list: list, list of  kmerse

    Returns
    ----------
    A: mat, matrix of atom to atom connections for each kmer; shape = (n_of_molecules, n_atoms, n_atoms)
    X mat, matrix of atom to features for each kmer; shape = (n_of_molecules, n_atoms, n_features)
    '''

    k = len(kmer_list[0])

    dna_base = {'A':'OP(=O)(O)OCC1OC(N3C=NC2=C(N)N=CN=C23)CC1',
            'T':'OP(=O)(O)OCC1OC(N2C(=O)NC(=O)C(C)=C2)CC1',
            'G':'OP(=O)(O)OCC1OC(N2C=NC3=C2N=C(N)N=C3O)CC1',
            'C':'OP(=O)(O)OCC1OC(N2C(=O)N=C(N)C=C2)CC1',
            'M':'OP(=O)(O)OCC1OC(N2C(=O)N=C(N)C(C)=C2)CC1'}

    rna_base = {'A':'OP(=O)(O)OCC1OC(N3C=NC2=C(N)N=CN=C23)C(O)C1',
                'T':'OP(=O)(O)OCC1OC(N2C(=O)NC(=O)C=C2)C(O)C1',
                'G':'OP(=O)(O)OCC1OC(N2C=NC3=C2N=C(N)N=C3O)C(O)C1',
                'C':'OP(=O)(O)OCC1OC(N2C(=O)N=C(N)C=C2)C(O)C1',
                'Q':'OP(=O)(O)OCC1OC(C2C(=O)NC(=O)NC=2)C(O)C1'}


    if n_type=="DNA":
        dna_smiles = kmer_chemistry.get_kmer_smiles(k, dna_base)
        dna_smiles = [dna_smiles.get(kmer)[0] for kmer in kmer_list]

        A, X = kmer_chemistry.get_AX_matrix(dna_smiles, ['C', 'N', 'O', 'P'], 133)

    elif n_type=="RNA":
        rna_smiles = kmer_chemistry.get_kmer_smiles(k, rna_base)
        rna_smiles = [rna_smiles.get(kmer)[0] for kmer in kmer_list]

        A, X = kmer_chemistry.get_AX_matrix(rna_smiles, ['C', 'N', 'O', 'P'], 116)
    return A,X

def fold_training(kmer_train,
                    kmer_test,
                    pA_train,
                    pA_test,
                    val_split=0):

    '''
    Function takes in train and test matrices and fits a new randomly initialized model.
    Function records training loss during training, validation (if selected), and also
    calculates pearson correlation, R2, and RMSE score on the test set

    Parameters
    ----------
    kmer_train: numpy mat; training matrix of kmers
    kmer_tedt: numpy mat; test matrix of kmers
    pA_train: numpy mat; training taget values of pA for kmers in kmer_train set
    pA_test: numpy mat; test target values of pA for kmers in kmer_test set
    val_split: int; percent of data to use as validation set during training

    Returns
    --------
    train_hist: dict; dictionary of training loss or validation loss if selected
    r: float; pearson correlation of predicted v. target values
    r2: float; R2 correlation of predicted v. target values
    rmse_score: float; RMSE score correlation of predicted v. target values

    '''
    # getting training and test A, X matrices, and their corresponding filters
    A_train, X_train = get_AX(kmer_train)
    gcn_filters_train = initialize_filters(A_train)
    A_test, X_test = get_AX(kmer_test)
    gcn_filters_test = initialize_filters(A_test)

    # initializing model - new randomly initialized model for every fold training
    if n_type=="DNA":
        model = initialize_model(X_train, gcn_filters_train, n_gcn=4, n_cnn=4, kernal_size_cnn=4, n_dense=4)
    elif n_type=="RNA":
        model = initialize_model(X_train, gcn_filters_train, n_gcn=1, n_cnn=4, kernal_size_cnn=4, n_dense=4)
    model.compile(loss='mean_squared_error', optimizer=Adam())

    # training model and testing performance
    train_hist = model.fit([X_train,gcn_filters_train],pA_train,validation_split=val_split, batch_size=128, epochs=350, verbose=verbosity)
    test_pred = model.predict([X_test, gcn_filters_test]).flatten()

    #calculating metrics
    r, _ = pearsonr(pA_test, test_pred)
    r2 = r2_score(pA_test, test_pred)
    rmse_score = rmse(test_pred, pA_test)

    return train_hist, r, r2, rmse_score

if __name__ == "__main__":
    ########----------------------Command line arguments--------------------##########
    parser = argparse.ArgumentParser(description="Script loads combined native and methylated kmers and performs 50-fold stratified cross validation")
    parser.add_argument('-o', '--OUT', default="out.npy", type=str, required=False, help='Full path for .npy file where results are saved')
    parser.add_argument('-v', '--VERBOSITY', default=0, type=int, required=False, help='Verbosity of model. Other than zero, loss per batch per epoch is printed. Default is 0, meaning nothing is printed')
    args=parser.parse_args()
    ########----------------------Command line arguments--------------------##########

    out = args.OUT

    global verbosity
    verbosity = args.VERBOSITY

    global n_type
    n_type = "DNA"

    local_out = str(os.environ['MYOUT']) # see job.yml for env definition

    all_kmers, all_pA, all_labels = cg_mg_combine()

    cv_res = {}

    for test_size, kmer_train_mat, kmer_test_mat,pA_train_mat,pA_test_mat in tqdm.tqdm(cv_folds(all_kmers,all_pA,labels = all_labels, folds=50),total=6):
        train_size = 1-test_size

        key = str(round(train_size,2))+'-'+str(round(test_size,2))

        cv_res[key] = {'r':[], 'r2':[],'rmse':[]}

        for i in range(kmer_train_mat.shape[0]):

            # each iteration is a fold
            kmer_train = kmer_train_mat[i]
            kmer_test = kmer_test_mat[i]
            pA_train = pA_train_mat[i]
            pA_test = pA_test_mat[i]

            train_hist, foldr, foldr2, fold_rmse = fold_training(kmer_train,kmer_test,pA_train,pA_test, val_split = 0.1)
            cv_res[key]['r'] += [foldr]
            cv_res[key]['r2'] += [foldr2]
            cv_res[key]['rmse'] += [(fold_rmse/kmer_test.shape[0])] #normalizing for number of samples in the test set.
            cv_res[key]['train_history']  = train_hist.history
            cv_res[key]['train_kmers'] = kmer_train
            cv_res[key]['test_kmers'] = kmer_test
            cv_res[key]['train_labels'] = pA_train
    
    np.save('.'+local_out+out, cv_res) #this will go to /results/