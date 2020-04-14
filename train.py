#! /usr/bin/env python
from keras.optimizers import Adam
from modules.nn_model import  initialize_model, initialize_filters,rmse
from modules.cv_utils import  kmer_parser
from modules.run import run_model
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from modules.kmer_chemistry import get_AX
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from keras.callbacks import EarlyStopping
from keras import backend as K
import tensorflow as tf

if __name__ == "__main__":
    ########----------------------Command line arguments--------------------##########
    parser = argparse.ArgumentParser(description="Run base-pair specific dropout cross validation")
    parser.add_argument('-i', '--FILE', default=None, type=str, required=False, help='kmer file with pA measurement')
    parser.add_argument('-model_fn', '--MODELFN', default="ndmi_model", type=str, required=False, help='name of model. To be used for saving the model and the weights')
    parser.add_argument('-o', '--RESULTS', type=str, default='out', required=False, help='Filename of results file')
    args=parser.parse_args()
    ########----------------------Command line arguments--------------------##########

    fn = args.FILE
    model_fn = args.MODELFN
    res_fn = args.RESULTS

    n_type = None
    if 'RNA' in fn or 'rna' in fn:
        n_type='rna'
    else:
        n_type="dna"

    kmer_list, pA_list = kmer_parser(fn)
    #kmer_train, kmer_test, pA_train, pA_test = train_test_split(kmer_list, pA_list, random_state=42, test_size=0.2)
    #print(kmer_train.shape)
    #print(kmer_test.shape)
    #print(pA_train.shape)
    #print(pA_test.shape)

    # getting adj and feature matrix for smiles
    A, X = get_AX(kmer_list)
    gcn_filters = initialize_filters(A)
    

    res_dict = {}
    print('running model')
    
    gpu_id = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)
    print("using gpu:", os.environ["CUDA_VISIBLE_DEVICES"])

    session = tf.Session(config=config)
    K.set_session(session)

    if n_type=="dna":
        model = initialize_model(X, gcn_filters, n_gcn=5, n_cnn=1, kernal_size_cnn=10, n_dense=5, dropout=0.1)
    elif n_type=="rna":
        model = initialize_model(X, gcn_filters, n_gcn=1, n_cnn=5, kernal_size_cnn=4, n_dense=5, dropout=0.1)
    model.compile(loss='mean_squared_error', optimizer=Adam())

    callbacks = [EarlyStopping(monitor='val_loss', min_delta=0.01, patience=25, verbose=1, mode='auto', baseline=None, restore_best_weights=False)]

    # training model and testing performance
    train_hist = model.fit([X,gcn_filters],pA_list,validation_split=0.1, batch_size=128, epochs=500, verbose=1, callbacks=callbacks)
    test_pred = model.predict([X, gcn_filters]).flatten()

    #calculating metrics
    r, _ = pearsonr(pA_list, test_pred)
    r2 = r2_score(pA_list, test_pred)
    rmse_score = rmse(pA_list, test_pred)

    
    res_dict['r'] = r
    res_dict['r2'] = r2
    res_dict['rmse'] = rmse_score
    res_dict['train_hist'] = train_hist
    res_dict['test_pred'] = test_pred

    model.to_hson()
    with open('./saved_models/%s.json'%model_fn,'r') as json_file:
        json_file.write(model_json)
    mode.save_weights('./saved_models/%s.h5'%model_fn)
    
    K.clear_session()
    
    np.save('./results/%s_%s.npy'%(n_type,res_fn), res_dict)
