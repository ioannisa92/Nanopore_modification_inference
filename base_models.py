#! /usr/bin/env python

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from modules.cv_utils import *
from sklearn.linear_model import LinearRegression
import argparse
import os
import tqdm
from sklearn.metrics import r2_score

def rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_pred - y_true)))


def fold_training(kmer_train,
                    kmer_test,
                    pA_train,
                    pA_test):

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

    Returns
    --------
    train_hist: dict; dictionary of training loss or validation loss if selected
    r: float; pearson correlation of predicted v. target values
    r2: float; R2 correlation of predicted v. target values
    rmse_score: float; RMSE score correlation of predicted v. target values

    '''
    # getting training and test A, X matrices, and their corresponding filters

    # initializing model - new randomly initialized model for every fold training
    model = LinearRegression(n_jobs=-1)
    

    # training model and testing performance
    model.fit(kmer_train, pA_train)
    test_pred = model.predict(kmer_test)
    
    #calculating metrics
    r, _ = pearsonr(pA_test.flatten(), test_pred.flatten())
    r2 = r2_score(pA_test.flatten(), test_pred.flatten())
    rmse_score = rmse(test_pred.flatten(), pA_test.flatten())

    return model, r, r2, rmse_score, test_pred


if __name__ == "__main__":

    ########----------------------Command line arguments--------------------##########
    parser = argparse.ArgumentParser(description="Script takes in a kmer and pA measurement file. The user can select between random cross validation, or targeted cross validation, where each based is hidden from each position of the kmer in training. Script saves cross validation resutls as a .npy file")

    parser.add_argument('-i', '--FILE', default=None, type=str, required=True, help='kmer file with pA measurement')
    parser.add_argument('-test_splits', '--SPLITS', nargs='+',type=float, required=False, default = np.arange(0.1,1,0.2), help='Test splits to run k-fold cross validation over')
    parser.add_argument('-k', '--FOLDS', type=int, default=50, required=False, help='K for fold numbers in cross validation')
    parser.add_argument('-o', '--OUT', default="out.npy", type=str, required=False, help='Full path for .npy file where results are saved')
    args=parser.parse_args()
    ########----------------------Command line arguments--------------------##########

    #fn = './ont_models/r9.4_180mv_450bps_6mer_DNA.model'
    #fn = './ont_models/r9.4_180mv_70bps_5mer_RNA.model'
    fn = args.FILE
    out = args.OUT
    test_splits = np.array(args.SPLITS)
    folds = args.FOLDS


    global n_type
    n_type = None
    if 'RNA' in fn or 'rna' in fn:
        n_type='RNA'
    else:
        n_type="DNA"


    local_out = "./results/"

    kmer_list, pA_list = kmer_parser_enc(fn)

    cv_res = {}
    
    
    for test_size, kmer_train_mat, kmer_test_mat,pA_train_mat,pA_test_mat in tqdm.tqdm(cv_folds(kmer_list,pA_list, folds=folds, test_sizes=test_splits),total=len(test_splits)):
        train_size = 1-test_size

        key = str(round(train_size,2))+'-'+str(round(test_size,2))

        cv_res[key] = {'r':[], 'r2':[],'rmse':[], "train_history":[],'train_kmers':[],'test_kmers':[], 'train_labels':[], 'test_labels':[], 'test_pred' : []}
        #for i in range(kmer_train_mat.shape[0]):
       

        for i in range(kmer_train_mat.shape[0]):
            # each iteration is a fold
            kmer_train = kmer_train_mat[i]
            kmer_test = kmer_test_mat[i]
            pA_train = pA_train_mat[i]
            pA_test = pA_test_mat[i]

            model, foldr, foldr2, fold_rmse, test_pred = fold_training(kmer_train,kmer_test,pA_train,pA_test)
            print(test_size, fold_rmse)

            cv_res[key]['r'] += [foldr]
            cv_res[key]['r2'] += [foldr2]
            cv_res[key]['rmse'] += [fold_rmse]

            cv_res[key]['train_kmers'] += [kmer_train]
            cv_res[key]['test_kmers'] += [kmer_test]
            cv_res[key]['train_labels'] += [pA_train]
            cv_res[key]['test_labels'] += [pA_test]
            cv_res[key]['test_pred'] += [test_pred] 
        
    np.save(local_out+out, cv_res) #this will go to /results/
