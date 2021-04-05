#!/usr/bin/env python3
from multiprocessing import Pool, Manager
from modules.nn_model import  get_available_gpus, initialize_model
from modules.run import run_model
import os
import numpy as np
import argparse
from modules.cv_utils import GPUGSCV, kmer_parser

def main():
    ########----------------------Command line arguments--------------------##########
    parser = argparse.ArgumentParser(description="Run gridsearch")
    parser.add_argument('-i', '--FILE', default=None, type=str, required=False, help='kmer file with pA measurement')
    parser.add_argument('-k', '--FOLDS', type=int, default=10, required=False, help='K for fold numbers in cross validation')
    args=parser.parse_args()
    ########----------------------Command line arguments--------------------##########
    
    fn = args.FILE
    k = args.FOLDS
    local_out = './results/'
    
    #fn = "./ont_models/r9.4_180mv_450bps_6mer_DNA.model"
    #fn = "./ont_models/r9.4_180mv_70bps_5mer_RNA.model"

    kmer_list, pA_list,label_list = kmer_parser(fn)
    print(label_list)
    all_bases = ''.join(kmer_list) 
    n_type = None
    if 'T' in all_bases and 'u' in all_bases:
        n_type = 'DNA_RNA'
    elif 'T' in all_bases and 'u' not in all_bases:
        n_type = 'DNA'
    elif 'T' not in all_bases and 'u'  in all_bases:
        n_type = 'RNA'
    elif 'T' not in all_bases and 'U'  in all_bases:
        n_type = 'RNA'

    print(n_type, flush=True)

 
    param_dict = dict(n_gcn=[2,3,4,5,6], 
                     n_cnn=[2,3,4,5,6], 
                     kernal_size_cnn=[2,4,10,20],
                     n_dense=[2, 4, 6,8,10],
                     dropout=[0.1])
    
    # for testing
    #param_dict = dict(n_gcn=[1],
    #                 n_cnn=[1],
    #                 kernal_size_cnn=[2],
    #                 n_dense=[1],
    #                 dropout=[0.1])
    
    try:
        n_gpus = get_available_gpus()
    except:
        n_gpus = 10
    
    gscv = GPUGSCV(initialize_model, param_dict, cv=k, n_gpus=n_gpus,res_fn='%s_gscv_results.npy'%n_type, n_type=n_type)

    gscv.fit(kmer_list, pA_list, labels=label_list)
    best_params = gscv.best_params
    cv_results = gscv.cv_results
    print(best_params, flush=True)
    #print(cv_results)
    print(gscv.best_score, flush=True)

    np.save('.'+local_out+'%s_best_params.npy'%(n_type.lower()), best_params) #this will go to /results/
    np.save('.'+local_out+'%s_gscv_results.npy'%(n_type.lower()), cv_results)

if __name__ == "__main__":
    main() 
