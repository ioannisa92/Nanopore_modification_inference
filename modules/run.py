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

def run_model(args, val_split=0,callbacks=True):

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
        print("using gpu:", os.environ["CUDA_VISIBLE_DEVICES"])

    # getting adj and feature matrix for smiles
    A_train, X_train = get_AX(kmer_train)
    gcn_filters_train = initialize_filters(A_train)
    A_test, X_test = get_AX(kmer_test)
    gcn_filters_test = initialize_filters(A_test)

    # configuring session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4 #what portion of gpu to use
    
    session = tf.Session(config=config)
    K.set_session(session)

    # initializing model - new randomly initialized model for every fold training
    if n_type=="DNA":
        model = initialize_model(X_train, gcn_filters_train, n_gcn=4, n_cnn=4, kernal_size_cnn=4, n_dense=4)
        model.compile(loss='mean_squared_error', optimizer=Adam())
    else: # for RNA
        print("RNA model")
        model = initialize_model(X_train, gcn_filters_train, n_gcn=1, n_cnn=4, kernal_size_cnn=4, n_dense=4)
        model.compile(loss='mean_squared_error', optimizer=Adam())

    if callbacks:
        callbacks = [EarlyStopping(monitor='loss', min_delta=0.01, patience=10, verbose=1, mode='min', baseline=None, restore_best_weights=False)]

    # training model and testing performance
    train_hist = model.fit([X_train,gcn_filters_train],pA_train,validation_split=val_split, batch_size=128, epochs=350, verbose=0, callbacks=callbacks)
    test_pred = model.predict([X_test, gcn_filters_test]).flatten()

    #calculating metrics
    r, _ = pearsonr(pA_test, test_pred)
    r2 = r2_score(pA_test, test_pred)
    rmse_score = rmse(test_pred, pA_test)

    # clearing session to avoid adding unwanted nodes on TF graph
    K.clear_session()
    avail_gpus.append(gpu_id)
    
    return r,r2,rmse_score, train_hist.history, kmer_train, kmer_test, pA_train, pA_test, test_pred

