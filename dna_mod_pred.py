#! /usr/bin/env python
from keras.optimizers import Adam
from modules.nn_model import  initialize_model, initialize_filters,rmse
from modules.cv_utils import  kmer_parser, gen_all_kmers
from modules.run import run_model
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from modules.kmer_chemistry import get_AX
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from keras.callbacks import EarlyStopping
from keras import backend as K
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from modules.utils import aws_upload
from sklearn.model_selection import ShuffleSplit

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
    local_out = str(os.environ['MYOUT']) # see job.yml for env definition

    #n_type = None
    #if 'RNA' in fn or 'rna' in fn:
    #    n_type='RNA'
    #else:
    #    n_type="DNA"
    
    print('parsing kmers...', flush=True) 
    mod_dna_kmer_fn = './ont_models/r9.4_450bps.cpg.m.only.6mer.template.model'
    
    kmer_list, pA_list,_ = kmer_parser(fn)
    kmer_mod_list, pA_mod_list = kmer_parser(mod_dna_kmer_fn)

    all_bases = ''.join(kmer_list)
    n_type = None
    if 'T' in all_bases and 'U' in all_bases:
        n_type = 'DNA_RNA'
    elif if 'T' in all_bases and 'U' not in all_bases:
        n_type = 'DNA'
    elif if 'T' not in all_bases and 'U'  in all_bases:
        n_type = 'RNA'

    print(n_type)



    #generating all kmers (with mods) that the model will predict pA on
    all_dna_mod_kmers = gen_all_kmers(alphabet=['A','T','C','G','M'], repeat=6)
    A_test_mod, X_test_mod = get_AX(all_dna_mod_kmers, n_type=n_type)
    gcn_filters_test_mod = initialize_filters(A_test_mod)


    res_dict = {}


    print('running model',flush=True)
    train_splits = np.arange(0.1,1,0.2)

    for train_split in train_splits:
    
        
        key = str(round(train_split,2)) # percent of modified kmers included in training
        print('running %s 50-fold...'%key,flush=True) 
        splitter = ShuffleSplit(n_splits=50, train_size=train_split, random_state=42).split(kmer_mod_list)
   
        res_dict[key] = {'train_kmers':[], 
                        'test_kmers':[],
                        'train_r':[],
                        "train_r2":[],
                        'train_rmse':[],
                        'train_hist':[], 
                        'test_pred':[], 
                        'train_pred':[]}
 
        for train_idx, _ in splitter:
    
            train_dna_mod_kmers = kmer_mod_list[train_idx]
            train_dna_mod_pA_list = pA_mod_list[train_idx]

            # including a train_split percentage modified kmers with canonical kmers
            train_kmers = np.concatenate([kmer_list, train_dna_mod_kmers],axis=0)
            pA_list_train = np.concatenate([pA_list, train_dna_mod_pA_list], axis=0)

            A_train, X_train = get_AX(train_kmers, n_type=n_type)
            gcn_filters_train = initialize_filters(A_train)
    
            gpu_id = 0
            os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)
            #print("using gpu:", os.environ["CUDA_VISIBLE_DEVICES"])

            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.8 #what portion of gpu to use

            session = tf.Session(config=config)
            K.set_session(session)

            if n_type=="DNA":
                model = initialize_model(X_train, gcn_filters_train, n_gcn=5, n_cnn=1, kernal_size_cnn=10, n_dense=5, dropout=0.1)
            elif n_type=="RNA":
                model = initialize_model(X_train, gcn_filters_train, n_gcn=1, n_cnn=5, kernal_size_cnn=4, n_dense=5, dropout=0.1)
            model.compile(loss='mean_squared_error', optimizer=Adam())

            callbacks = [EarlyStopping(monitor='loss', min_delta=0.01, patience=10, verbose=1, mode='auto', baseline=None, restore_best_weights=False)]

            # training model and testing performance
            train_hist = model.fit([X_train,gcn_filters_train],pA_list_train,validation_split=0, batch_size=128, epochs=500, verbose=0, callbacks=callbacks)
            train_pred = model.predict([X_train, gcn_filters_train]).flatten()
            test_pred = model.predict([X_test_mod, gcn_filters_test_mod]).flatten()

            #calculating metrics
            r, _ = pearsonr(pA_list_train, train_pred)
            r2 = r2_score(pA_list_train, train_pred)
            rmse_score = rmse(pA_list_train, train_pred)

            res_dict[key]['train_kmers'] += [train_kmers]
            res_dict[key]['test_kmers'] += [all_dna_mod_kmers]
            res_dict[key]['train_r'] += [r]
            res_dict[key]['train_r2'] += [r2]
            res_dict[key]['train_rmse'] += [rmse_score]
            res_dict[key]['train_hist'] += [train_hist.history]
            res_dict[key]['test_pred'] += [test_pred]
            res_dict[key]['train_pred'] += [train_pred]

            K.clear_session()
    
             
    np.save('.'+local_out+'%s.npy'%(res_fn), res_dict)
    aws_upload('.'+local_out+'%s.npy'%(res_fn))
