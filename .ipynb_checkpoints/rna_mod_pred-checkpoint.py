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
import tensorflow as tf
import os
from modules.utils import aws_upload

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

    kmer_list, pA_list,_ = kmer_parser(fn)

    all_bases = ''.join(kmer_list)
    n_type = None
    if 'T' in all_bases and 'U' in all_bases:
        n_type = 'DNA_RNA'
    elif 'T' in all_bases and 'U' not in all_bases:
        n_type = 'DNA'
    elif 'T' not in all_bases and 'U'  in all_bases:
        n_type = 'RNA'

    print(n_type, flush=True)


    # getting adj and feature matrix for smiles
    A, X = get_AX(kmer_list, n_type=n_type)
    gcn_filters = initialize_filters(A)

    #generating all kmers (with mods) that the model will predict pA on
    rna_kmers = gen_all_kmers(alphabet=['A','U','C','G','X','Y','Z'], repeat=5)
    A_test, X_test = get_AX(rna_kmers, n_type=n_type)
    gcn_filters_test = initialize_filters(A_test)

    res_dict = {}
    res_dict['train_kmers'] = []
    res_dict['test_kmers'] = []
    res_dict['train_r'] = []
    res_dict['train_r2'] = []
    res_dict['train_rmse'] = []
    res_dict['train_hist'] = []
    res_dict['test_pred'] = []
    res_dict['train_pred'] = []

    best_score=None
    best_repeat = None

    prp = str(os.environ['PRP'])
    local_out = str(os.environ['MYOUT'])
    s3out = str(os.environ['S3OUT'])


    print('running model')
    repeat = 50
    epochs = 500
    for i in range(repeat): 
        gpu_id = 0
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)
        print("using gpu:", os.environ["CUDA_VISIBLE_DEVICES"])

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.8 #what portion of gpu to use

        session = tf.Session(config=config)
        K.set_session(session)

        if n_type=="DNA":
            model = initialize_model(X, gcn_filters, n_gcn=5, n_cnn=1, kernal_size_cnn=10, n_dense=5, dropout=0.1)
        elif n_type=="RNA":
            model = initialize_model(X, gcn_filters, n_gcn=6, n_cnn=6, kernal_size_cnn=10, n_dense=10, dropout=0.1)
        model.compile(loss='mean_squared_error', optimizer=Adam())

        callbacks = [EarlyStopping(monitor='loss', min_delta=0.01, patience=10, verbose=1, mode='auto', baseline=None, restore_best_weights=False)]

        # training model and testing performance
        train_hist = model.fit([X,gcn_filters],pA_list,validation_split=0, batch_size=128, epochs=epochs, verbose=1, callbacks=callbacks)
        train_pred = model.predict([X, gcn_filters]).flatten()
        test_pred = model.predict([X_test, gcn_filters_test]).flatten()

        #calculating metrics
        r, _ = pearsonr(pA_list, train_pred)
        r2 = r2_score(pA_list, train_pred)
        rmse_score = rmse(pA_list, train_pred)

        res_dict['train_kmers'] += [kmer_list]
        res_dict['test_kmers'] += [rna_kmers]
        res_dict['train_r'] += [r]
        res_dict['train_r2'] += [r2]
        res_dict['train_rmse'] += [rmse_score]
        res_dict['train_hist'] += [train_hist.history]
        res_dict['test_pred'] += [test_pred]
        res_dict['train_pred'] += [train_pred]

        #model_json = model.to_json()
        #with open('.'+local_out+'%s.json'%model_fn,'w') as json_file:
        #    json_file.write(model_json)

        #model.save_weights('./results/%s_repeat%d.h5'%(model_fn,i))

        #aws_upload('./results/%s_repeat%d.h5'%(model_fn,i))
        #aws_upload('.'+local_out+'%s.json'%model_fn)

        #session = boto3.session.Session(profile_name="default")
        #bucket = session.resource("s3", endpoint_url=prp).Bucket("stuartlab")
        #bucket.upload_file('./results/%s_repeat%d.h5'%(model_fn,i), s3out+'%s_repeat%d.h5'%(model_fn,i))
        #bucket.upload_file('.'+local_out+'%s.json'%model_fn, s3out+'%s.json'%model_fn)
        K.clear_session()
    
    #optimal_model= '%s_repeat_%d.h5'%(model_fn,best_repeat)
    #with open('temp.txt','w') as f:
    #    f.write(optimal_model)
    #print('set env var',os.environ['OPTMODEL'])
    #for fn in os.listdir('.'+local_out):
    #    if '%s_repeat'%model_fn in fn:
    #        if fn != optimal_model:
    #            os.remove('.'+local_out+fn)
                
             
    np.save('.'+local_out+'%s.npy'%(res_fn), res_dict)
    aws_upload('.'+local_out+'%s.npy'%(res_fn))
