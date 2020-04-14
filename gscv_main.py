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
    parser = argparse.ArgumentParser(description="Run base-pair specific dropout cross validation")
    parser.add_argument('-i', '--FILE', default=None, type=str, required=False, help='kmer file with pA measurement')
    parser.add_argument('-k', '--FOLDS', type=int, default=10, required=False, help='K for fold numbers in cross validation')
    args=parser.parse_args()
    ########----------------------Command line arguments--------------------##########
    
    fn = args.FILE
    k = args.FOLDS
    local_out = str(os.environ['MYOUT']) # see job.yml for env definition

    n_type = None
    if 'RNA' in fn or 'rna' in fn:
        n_type='rna'
    else:
        n_type="dna"
    
    #fn = "./ont_models/r9.4_180mv_450bps_6mer_DNA.model"
    #fn = "./ont_models/r9.4_180mv_70bps_5mer_RNA.model"

    kmer_list, pA_list = kmer_parser(fn)
    param_dict = dict(n_gcn=[1,2,3,4,5], 
                     n_cnn=[1,2,3,4,5], 
                     kernal_size_cnn=[2,4,10,20],
                     n_dense=[1,2,3,4,5],
                     dropout=[0.1,0.2])
    
    # for testing
    #param_dict = dict(n_gcn=[1,2],
    #                 n_cnn=[1,2],
    #                 kernal_size_cnn=[2,4],
    #                 n_dense=[1,2],
    #                 dropout=[0.1,0.2])
    
    try:
        n_gpus = get_available_gpus()
    except:
        n_gpus = 10
    
    gscv = GPUGSCV(initialize_model, param_dict, cv=k, n_gpus=n_gpus,res_fn='%s_gscv_results.npy'%n_type)

    gscv.fit(kmer_list, pA_list)
    best_params = gscv.best_params
    cv_results = gscv.cv_results
    print(best_params)
    print(cv_results)
    print(gscv.best_score)

    np.save('.'+local_out+'%s_best_params.npy'%n_type, best_params) #this will go to /results/
    np.save('.'+local_out+'%s_gscv_results.npy'%n_type, cv_results)

if __name__ == "__main__":
    main() 