#! /usr/bin/env python3
from modules.nn_model import rmse, initialize_filters
from modules.cv_utils import kmer_parser,gen_all_kmers
import numpy as np
import argparse
from modules.kmer_chemistry import get_AX
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from keras import backend as K
import tensorflow as tf
import os
from keras.models import model_from_json
from modules.spectral_layers import MultiGraphCNN

if __name__ == "__main__":
    ########----------------------Command line arguments--------------------##########
    parser = argparse.ArgumentParser(description="Run base-pair specific dropout cross validation")
    parser.add_argument('-i', '--FILE', default=None, type=str, required=False, help='kmer file with pA measurement')
    parser.add_argument('-model_path', '--MODELPATH', default="ndmi_model", type=str, required=False, help='PATH of models json file. To be used for loading the model and the weights')
    parser.add_argument('-o', '--RESULTS', type=str, default='out', required=False, help='Filename of results file')
    args=parser.parse_args()
    ########----------------------Command line arguments--------------------##########

    fn = args.FILE
    model_path = args.MODELPATH
    model_path = '.'+''.join(model_path.split('.')[:-1])
    res_fn = args.RESULTS


    #kmer_list, pA_list = kmer_parser(fn)
    kmer_list = gen_all_kmers()

    # getting adj and feature matrix for smiles
    A, X = get_AX(kmer_list)
    gcn_filters = initialize_filters(A)


    res_dict = {}
    #res_dict['r'] = []
    #res_dict['r2'] = []
    #res_dict['rmse'] = []
    res_dict['prediction'] = []
    res_dict['kmers'] = kmer_list
    print('running model')

    for i in range(50):
        print('running repeat %d'%i)
        gpu_id = 0
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)
        print("using gpu:", os.environ["CUDA_VISIBLE_DEVICES"])

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.8 #what portion of gpu to use

        session = tf.Session(config=config)
        K.set_session(session)
        with open(model_path+'.json', 'r') as json_file:
            architecture = json_file.read()
            model = model_from_json(architecture, custom_objects={'MultiGraphCNN':MultiGraphCNN})

            model.load_weights(model_path+'_repeat%d.h5'%i)

        prediction = model.predict([X,gcn_filters]).flatten()
    
        #calculating metrics
        #r, _ = pearsonr(pA_list, prediction)
        #r2 = r2_score(pA_list, prediction)
        #rmse_score = rmse(pA_list, prediction)

        #res_dict['r'] += [r]
        #res_dict['r2'] += [r2]
        #res_dict['rmse'] += [rmse_score]
        res_dict['prediction'] += [prediction]
        #res_dict['ground_truth'] += [pA_list]
    
        K.clear_session()
    
    np.save(res_fn, res_dict) 
