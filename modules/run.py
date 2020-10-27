import os
import numpy as np
from .nn_model import initialize_model, initialize_filters,rmse
from keras.callbacks import EarlyStopping
from keras import backend as K
import tensorflow as tf
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from .kmer_chemistry import get_AX
from keras.optimizers import Adam

def run_model(args, val_split=0.1):

    '''
    Function takes in train and test matrices and fits a new randomly initialized model.
    Function records training loss during training, validation (if selected), and also
    calculates pearson correlation, R2, and RMSE score on the test set

    Parameters
    ----------
    args : tuple containing
        kmer_train: numpy mat
             training matrix of kmers
        kmer_test: numpy mat
            test matrix of kmers
        pA_train: numpy mat
            training taget values of pA for kmers in kmer_train set
        pA_test: numpy mat
            test target values of pA for kmers in kmer_test set
        n_type: str
            whether the model loaded should be for DNA or RNA
        avail_gpus: list
            list of available gpus indeces
    val_split: float
        percent of data to use as validation set during training

    callbacks: bool
        whether to use callbacks in training. callback used will be EarlyStopping

    Returns
    --------
    train_hist: dict
        dictionary of training loss or validation loss if selected
    r: float
        pearson correlation of predicted v. target values
    r2: float
        R2 correlation of predicted v. target values
    rmse_score: float
        RMSE score correlation of predicted v. target values

    '''
    # parse tuple args
    kmer_train, pA_train =args[0], args[1]
    kmer_test, pA_test = args[2], args[3]
    n_type = args[4]
    avail_gpus = args[5]

    if len(avail_gpus)!=0: # in case no gpu_id is passed, cpu will be used
        # read the available GPU for training
        gpu_id = avail_gpus.pop(0)
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)
        print("using gpu:", os.environ["CUDA_VISIBLE_DEVICES"], flush=True)

    # getting adj and feature matrix for smiles
    A_train, X_train = get_AX(kmer_train, n_type=n_type)
    gcn_filters_train = initialize_filters(A_train)
    A_test, X_test = get_AX(kmer_test, n_type=n_type)
    gcn_filters_test = initialize_filters(A_test)

    # configuring session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8 #what portion of gpu to use
    
    session = tf.Session(config=config)
    K.set_session(session)

    if n_type=="DNA":
        model = initialize_model(X_train, gcn_filters_train, n_gcn=4, n_cnn=3, kernal_size_cnn=10, n_dense=10, dropout=0.1)
    elif n_type=="RNA":
        model = initialize_model(X_train, gcn_filters_train, n_gcn=4, n_cnn=5, kernal_size_cnn=10, n_dense=10, dropout=0.1)
    #elif n_type == 'DNA_RNA':
    #    model = initialize_model(X_train, gcn_filters_train, n_gcn=4, n_cnn=1, kernal_size_cnn=10, n_dense=5, dropout=0.1)
    model.compile(loss='mean_squared_error', optimizer=Adam())

    callbacks = [EarlyStopping(monitor='val_loss', min_delta=0.01, patience=10, verbose=1, mode='auto', baseline=None, restore_best_weights=False)]

    # training model and testing performance
    train_hist = model.fit([X_train,gcn_filters_train],pA_train,validation_split=val_split, batch_size=128, epochs=500, verbose=0, callbacks=callbacks)
    test_pred = model.predict([X_test, gcn_filters_test]).flatten()
    train_pred = model.predict([X_train, gcn_filters_train]).flatten()

    #calculating metrics
    r, _ = pearsonr(pA_test, test_pred)
    r2 = r2_score(pA_test, test_pred)
    rmse_score = rmse(test_pred, pA_test)

    # clearing session to avoid adding unwanted nodes on TF graph
    K.clear_session()
    avail_gpus.append(gpu_id)
    
    return r,r2,rmse_score, train_hist.history, kmer_train, kmer_test, pA_train, pA_test, test_pred, train_pred




def run_params(args):

    '''
    Function takes in train and test matrices and fits a new randomly initialized model.
    Function records training loss during training, validation (if selected), and also
    calculates pearson correlation, R2, and RMSE score on the test set

    Parameters
    ----------
    args : tuple containing
        model : function
            Model function with the params that are needed to be tested
        kmer_train: numpy mat
             training matrix of kmers
        kmer_test: numpy mat
            test matrix of kmers
        pA_train: numpy mat
            training taget values of pA for kmers in kmer_train set
        pA_test: numpy mat
            test target values of pA for kmers in kmer_test set
        n_type: str
            whether the model loaded should be for DNA or RNA
        avail_gpus: list
            list of available gpus indeces
    val_split: float
        percent of data to use as validation set during training

    callbacks: bool
        whether to use callbacks in training. callback used will be EarlyStopping

    Returns
    --------
    train_hist: dict
        dictionary of training loss or validation loss if selected
    r: float
        pearson correlation of predicted v. target values
    r2: float
        R2 correlation of predicted v. target values
    rmse_score: float
        RMSE score correlation of predicted v. target values

    '''
    # parse tuple args
    model = args[0] # model loaded with params to test
    pA_train = args[1]
    pA_test = args[2]
    pA_valid = args[3]
    X_train, gcn_filters_train = args[4], args[5]
    X_test, gcn_filters_test = args[6], args[7]
    X_valid, gcn_filters_valid = args[8], args[9]
    avail_gpus = args[10]

    if len(avail_gpus)!=0: # in case no gpu_id is passed, cpu will be used
        # read the available GPU for training
        gpu_id = avail_gpus.pop(0)
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)
        print("using gpu:", os.environ["CUDA_VISIBLE_DEVICES"], flush=True)


    # configuring session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9 #what portion of gpu to use
    #config.gpu_options.allow_growth = True
    
    session = tf.Session(config=config)
    K.set_session(session)

    callbacks = [EarlyStopping(monitor='val_loss', min_delta=0.01, patience=15, verbose=1, mode='auto', baseline=None, restore_best_weights=False)]

    # initializing model - new randomly initialized model for every fold training
    model.compile(loss='mean_squared_error', optimizer=Adam())

    # training model and testing performance
    train_hist = model.fit([X_train,gcn_filters_train],pA_train,validation_data=([X_valid,gcn_filters_valid],pA_valid), batch_size=128, epochs=500, verbose=0, callbacks=callbacks)
    test_pred = model.predict([X_test, gcn_filters_test]).flatten()
    train_pred = model.predict([X_train, gcn_filters_train]).flatten()
    
    avail_gpus.append(gpu_id)

    #calculating metrics
    r, _ = pearsonr(pA_test.flatten(), test_pred)
    r2 = r2_score(pA_test.flatten(), test_pred)
    rmse_score = rmse(test_pred.flatten(), pA_test)

    # clearing session to avoid adding unwanted nodes on TF graph
    K.clear_session()
    
    return r,r2,rmse_score, train_hist.history, test_pred, train_pred



