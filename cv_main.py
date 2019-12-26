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

    base = {'A':'OP(=O)(O)OCC1OC(N3C=NC2=C(N)N=CN=C23)CC1',
            'T':'OP(=O)(O)OCC1OC(N2C(=O)NC(=O)C(C)=C2)CC1',
            'G':'OP(=O)(O)OCC1OC(N2C=NC3=C2N=C(N)N=C3O)CC1',
            'C':'OP(=O)(O)OCC1OC(N2C(=O)N=C(N)C=C2)CC1',
            'M':'OP(=O)(O)OCC1OC(N2C(=O)N=C(N)C(C)=C2)CC1'}

    dna_smiles = kmer_chemistry.get_kmer_smiles(6, base)

    smiles = [dna_smiles.get(kmer)[0] for kmer in kmer_list]
    A, X = kmer_chemistry.get_AX_matrix(smiles, ['C', 'N', 'O', 'P'], 133)
    
    return A,X 

def fold_training(kmer_train,
                    kmer_test,
                    pA_train,
                    pA_test):
    

    # getting training and test A, X matrices, and their corresponding filters
    A_train, X_train = get_AX(kmer_train)
    gcn_filters_train = initialize_filters(A_train)
    A_test, X_test = get_AX(kmer_test)
    gcn_filters_test = initialize_filters(A_test)
    
    # initializing model
    model = initialize_model(X_train, gcn_filters_train, n_gcn=4, n_cnn=4, kernal_size_cnn=4, n_dense=4)
    model.compile(loss='mean_squared_error', optimizer=Adam())
 
    # training model and testing performance
    train_hist = model.fit([X_train,gcn_filters_train],pA_train, batch_size=128, epochs=500, verbose=1)
    test_pred = model.predict([X_test, gcn_filters_test]).flatten()
    
    #calculating metrics
    r, _ = stats.pearsonr(pA_test, test_pred)
    r2 = r2_score(pas_test, test_pred)
    rmse(test_pred, pA_test)
    
    return r, r2, rmse

if __name__ == "__main__":

   ########----------------------Command line arguments--------------------##########
    parser = argparse.ArgumentParser(description="Arguments for preranked an single sample GSEA")

    parser.add_argument('-i', '--FILE', default=None, type=str, required=False, help='kmer file with pA measurement')
    parser.add_argument('-cv', '--CV', required=False, action='store_true',help='Random CV splits of variable size')
    parser.add_argument('-kmer_cv', '--KMERCV', required=False, action='store_true',help='CV splits based on position of base')

    args=parser.parse_args()
    ########----------------------Command line arguments--------------------########## 
    
    fn = './ont_models/r9.4_180mv_450bps_6mer_DNA.model'
    #fn = ars.FILE
    cv = args.CV
    kmer_cv = args.KMERCV

    kmer_list, pA_list = kmer_parser(fn)

    if cv:

        cv_res = {}
    
        for test_size, kmer_train_mat,kmer_test_mat,pA_train_mat,pA_test_mat in cv_folds(kmer_list,pA_list):
            train_size = 1-test_size
    
            key = str(train_size)+'-'+str(test_size)
        
            cv_res[key] = {'r':[], 'r2':[],'rmse':[]}
 
            for i in range(kmer_train_mat.shape[0]):
                
                # each iteration is a fold
                kmer_train = kmer_train_mat[i]
                kmer_test = kmer_test_mat[i]
                pA_train = pA_train_mat[i]
                pA_test = pA_test_mat[i]
                
                foldr, foldr2, fold_rmse = fold_training(kmer_train,kmer_test,pA_train,pA_test)
                cv_res[key]['r'] += [foldr]
                cv_res[key]['r2'] += [foldr2]
                cv_res[key]['rmse'] += [fold_rmse]                 

        np.save('./results/cv_results.npy', cv_res)

    if kmer_cv:
        
        kmer_cv_res = {}
        base_order = ['A','T','C','G'] # order of bases in matrices below
    
        for pos, kmer_train_mat,kmer_test_mat,pA_train_mat,pA_test_mat in base_folds(kmer_list,pA_list):
            key = 'Pos%d'%pos

            for i in range(kmer_train_mat.shape[0]):
                base_examined = base_order[i]
                key = key + '-' + base_examined
                
                kmer_cv_res[key] = {'r':None, 'r2':None,'rmse':None}
            
                # each iteration is a fold
                kmer_train = kmer_train_mat[i]
                kmer_test = kmer_test_mat[i]
                pA_train = pA_train_mat[i]
                pA_test = pA_test_mat[i]

                foldr, foldr2, fold_rmse = fold_training(kmer_train,kmer_test,pA_train,pA_test)
                kmer_cv_res[key]['r'] = foldr
                kmer_cv_res[key]['r2'] = foldr2
                kmer_cv_res[key]['rmse'] = fold_rmse
        
        np.save('./results/kmer_cv_results.npy', kmer_cv_res)
