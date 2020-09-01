#! /usr/bin/env python

import argparse
from modules import kmer_chemistry
from modules.nn_model import *
from modules.cv_utils import *
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import tqdm
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping
#imports

def rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_pred - y_true)))

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
    callbacks: bool; whether to use callbacks in training

    Returns
    --------
    train_hist: dict; dictionary of training loss or validation loss if selected
    r: float; pearson correlation of predicted v. target values
    r2: float; R2 correlation of predicted v. target values
    rmse_score: float; RMSE score correlation of predicted v. target values
    
    '''
    # getting training and test A, X matrices, and their corresponding filters
    A_train, X_train = kmer_chemistry.get_AX(kmer_train,n_type=n_type)
    gcn_filters_train = initialize_filters(A_train)
    A_test, X_test = kmer_chemistry.get_AX(kmer_test,n_type=n_type)
    gcn_filters_test = initialize_filters(A_test)
   
    gpu_id = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)
    print("using gpu:", os.environ["CUDA_VISIBLE_DEVICES"])

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8 #what portion of gpu to use

    session = tf.Session(config=config)
    K.set_session(session)
 
    # initializing model - new randomly initialized model for every fold training
    if n_type=="DNA":
        model = initialize_model(X_train, gcn_filters_train, n_gcn=5, n_cnn=1, kernal_size_cnn=10, n_dense=5, dropout=0.1)
    elif n_type=="RNA":
        model = initialize_model(X_train, gcn_filters_train, n_gcn=1, n_cnn=5, kernal_size_cnn=4, n_dense=5, dropout=0.1)
    model.compile(loss='mean_squared_error', optimizer=Adam())

    callbacks = [EarlyStopping(monitor='val_loss', min_delta=0.01, patience=10, verbose=1, mode='auto', baseline=None, restore_best_weights=False)]
 
    # training model and testing performance
    train_hist = model.fit([X_train,gcn_filters_train],pA_train,validation_split=val_split, batch_size=128, epochs=500, verbose=verbosity, callbacks=callbacks)
    test_pred = model.predict([X_test, gcn_filters_test]).flatten()
    train_pred = model.predict([X_train, gcn_filters_train]).flatten()
 
    #calculating metrics
    r, _ = pearsonr(pA_test, test_pred)
    r2 = r2_score(pA_test, test_pred)
    rmse_score = rmse(test_pred, pA_test)

    # clearing session to avoid adding unwanted nodes on TF graph
    K.clear_session()
    return train_hist, r, r2, rmse_score, test_pred, train_pred



if __name__ == "__main__":

    ########----------------------Command line arguments--------------------##########
    parser = argparse.ArgumentParser(description="Script takes in a kmer and pA measurement file. The user can select between random cross validation, or targeted cross validation, where each based is hidden from each position of the kmer in training. Script saves cross validation resutls as a .npy file")

    parser.add_argument('-i', '--FILE', default=None, type=str, required=False, help='kmer file with pA measurement')
    parser.add_argument('-cv', '--CV', required=False, action='store_true',help='MODE: Random CV splits of variable size')
    parser.add_argument('-kmer_cv', '--KMERCV', required=False, action='store_true',help='MODE: CV splits based on position of base')
    parser.add_argument('-test_splits', '--SPLITS', nargs='+',type=float, required=False, default = np.arange(0.05,1,0.05), help='Test splits to run k-fold cross validation over')
    parser.add_argument('-k', '--FOLDS', type=int, default=50, required=False, help='K for fold numbers in cross validation')
    parser.add_argument('-o', '--OUT', default="out.npy", type=str, required=False, help='Full path for .npy file where results are saved')
    parser.add_argument('-v', '--VERBOSITY', default=0, type=int, required=False, help='Verbosity of model. Other than zero, loss per batch per epoch is printed. Default is 0, meaning nothing is printed')
    args=parser.parse_args()
    ########----------------------Command line arguments--------------------########## 
    
    #fn = './ont_models/r9.4_180mv_450bps_6mer_DNA.model'
    #fn = './ont_models/r9.4_180mv_70bps_5mer_RNA.model'
    fn = args.FILE
    cv = args.CV
    kmer_cv = args.KMERCV
    out = args.OUT
    test_splits = np.array(args.SPLITS)
    folds = args.FOLDS
    
    global verbosity
    verbosity = args.VERBOSITY  

    local_out = str(os.environ['MYOUT']) # see job.yml for env definition

    kmer_list, pA_list,_ = kmer_parser(fn)
    all_bases = ''.join(list(kmer_list))

    global n_type
    n_type = None
    if 'T' in all_bases and 'U' in all_bases:
        n_type = 'DNA_RNA'
    elif 'T' in all_bases and 'U' not in all_bases:
        n_type = 'DNA'
    elif 'T' not in all_bases and 'U'  in all_bases:
        n_type = 'RNA'

    print(n_type)


    if cv:
        
        cv_res = {}
    
        for test_size, kmer_train_mat, kmer_test_mat,pA_train_mat,pA_test_mat in tqdm.tqdm(cv_folds(kmer_list,pA_list, folds=folds, test_sizes=test_splits),total=len(test_splits)):
            train_size = 1-test_size
    
            key = str(round(train_size,2))+'-'+str(round(test_size,2))
        
            cv_res[key] = {'r':[], 'r2':[],'rmse':[], "train_history":[],'train_kmers':[],'test_kmers':[], 'train_labels':[], 'test_labels':[], 'test_pred' : [],'train_pred':[]}
 
            for i in range(kmer_train_mat.shape[0]):
                
                # each iteration is a fold
                kmer_train = kmer_train_mat[i]
                kmer_test = kmer_test_mat[i]
                pA_train = pA_train_mat[i]
                pA_test = pA_test_mat[i]
                
                train_hist, foldr, foldr2, fold_rmse, test_pred,train_pred = fold_training(kmer_train,kmer_test,pA_train,pA_test, val_split = 0.1)
                cv_res[key]['r'] += [foldr]
                cv_res[key]['r2'] += [foldr2]
                cv_res[key]['rmse'] += [fold_rmse] 

                cv_res[key]['train_history']  += [train_hist.history]        
                cv_res[key]['train_kmers'] += [kmer_train]
                cv_res[key]['test_kmers'] += [kmer_test]
                cv_res[key]['train_labels'] += [pA_train]
                cv_res[key]['test_labels'] += [pA_test]
                cv_res[key]['test_pred'] += [test_pred]   
                cv_res[key]['train_pred'] += [train_pred]

        np.save('.'+local_out+out, cv_res) #this will go to /results/


    if kmer_cv:
        
        kmer_cv_res = {}
        base_order = ['A','T','C','G'] # order of bases in matrices below
    
        for pos, kmer_train_mat,kmer_test_mat,pA_train_mat,pA_test_mat in tqdm.tqdm(base_folds(kmer_list,pA_list),total=7):
            key = 'Pos%d'%pos

            for i in range(kmer_train_mat.shape[0]):
                base_examined = base_order[i]
                key_ = key + '-' + base_examined
                
                #kmer_cv_res[key_] = {'r':None, 'r2':None,'rmse':None}
                kmer_cv_res[key_] = {'r':[], 'r2':[],'rmse':[], "train_history":[],'train_kmers':[],'test_kmers':[], 'train_labels':[], 'test_labels':[], 'test_pred' : [],'train_pred':[]}

                # each iteration is a fold
                kmer_train = kmer_train_mat[i]
                kmer_test = kmer_test_mat[i]
                pA_train = pA_train_mat[i]
                pA_test = pA_test_mat[i]
                
                for i in np.arange(50):
                    train_hist, foldr, foldr2, fold_rmse, test_pred,train_pred = fold_training(kmer_train,kmer_test,pA_train,pA_test, val_split = 0.1)
                    kmer_cv_res[key_]['r'] += [foldr]
                    kmer_cv_res[key_]['r2'] += [foldr2]
                    kmer_cv_res[key_]['rmse'] += [fold_rmse] 
                    
                    kmer_cv_res[key_]['train_history']  += [train_hist.history]
                    kmer_cv_res[key_]['train_kmers'] += [kmer_train]
                    kmer_cv_res[key_]['test_kmers'] += [kmer_test]
                    kmer_cv_res[key_]['train_labels'] += [pA_train]
                    kmer_cv_res[key_]['test_pred'] += [test_pred]
                    kmer_cv_res[key_]['test_labels'] += [pA_test]
                    kmer_cv_res[key_]['train_pred'] += [train_pred]
                
        np.save('.'+local_out+out, kmer_cv_res)
