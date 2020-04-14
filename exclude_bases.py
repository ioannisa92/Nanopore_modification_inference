#!/usr/bin/env python3
from multiprocessing import Pool, Manager
from modules.cv_utils import kmer_parser
from modules.nn_model import  get_available_gpus
from modules.run import run_model
import os
import numpy as np
import argparse

class GetTrainTest():
    
    def __init__(self, fn, base_pair):
        self.fn = fn
        self.base_pair = base_pair
        self.all_kmers, self.all_pAs = kmer_parser(self.fn) # all kmer
        self.all_kmers_dict = dict(zip(self.all_kmers, self.all_pAs))
        
        self.train_kmers = []
        for base in self.base_pair:
            self.kmers_minus_base, _ = kmer_parser(self.fn, exclude_base=base)
            self.train_kmers += [self.kmers_minus_base]
    
    def get_train(self):
        self.train_kmers = set().union(*self.train_kmers)
        train_pa = []
        for train_kmer in self.train_kmers:
            train_pa += [self.all_kmers_dict[train_kmer]]
        
        self.train_pa = np.array(train_pa)
        return np.array(list(self.train_kmers)), np.array(train_pa)
    
    def get_test(self):
        self.test_kmers = set(self.all_kmers) - set(self.train_kmers)
        test_pa = []
        for test_kmer in self.test_kmers:
            test_pa +=[self.all_kmers_dict[test_kmer]]
        self.test_pa = np.array(test_pa)
        return np.array(list(self.test_kmers)), np.array(self.test_pa)  

    def get_train_test(self):
        train_kmers, train_pa = self.get_train()
        test_kmers, test_pa = self.get_test()
        return train_kmers, train_pa, test_kmers, test_pa

def wrap(args):
    print("in wrap")
    bases = args[0] # tuple
    if len(bases) >= 2:
        key = '-'.join(list(bases))
    elif len(bases) == 1:
        key=bases[0]
    avail_gpus = args[1]
    res_dict = args[2]
    fn = args[3]
    n_type=args[4]
    
    c = GetTrainTest(fn, bases)
    train_kmers, train_pa = c.get_train()
    test_kmers, test_pa = c.get_test()
    
    for i in np.arange(50): 
        r,r2,rmse_score, train_hist, kmer_train, kmer_test, pA_train, pA_test, test_pred, train_pred = run_model((train_kmers, train_pa, test_kmers, test_pa, n_type, avail_gpus))

        res_dict[key]['r'] += [r]
        res_dict[key]['r2'] += [r2]
        res_dict[key]['rmse'] += [rmse_score]

        res_dict[key]['train_history']  += [train_hist]
        res_dict[key]['train_kmers'] += [kmer_train]
        res_dict[key]['test_kmers'] += [kmer_test]
        res_dict[key]['train_labels'] += [pA_train]
        res_dict[key]['test_labels'] += [pA_test]
        res_dict[key]['test_pred'] += [test_pred]
        res_dict[key]['train_pred'] += [train_pred]
        
        print("dict updated")

def main():
    ########----------------------Command line arguments--------------------##########
    parser = argparse.ArgumentParser(description="Run base-pair specific dropout cross validation")
    parser.add_argument('-i', '--FILE', default=None, type=str, required=False, help='kmer file with pA measurement')
    parser.add_argument('-base_pair_exclude', '--PAIRS', required=False, action='store_true', help='MODE: pairs of pases will be excluded')
    parser.add_argument('-base_exclude', '--SOLO', required=False, action = 'store_true', help='MODE: each of the four bases will be removed from training')
    parser.add_argument('-o', '--OUT', default="out.npy", type=str, required=False, help='Full path for .npy file where results are saved')
    args=parser.parse_args()
    ########----------------------Command line arguments--------------------##########
    fn = args.FILE
    out = args.OUT
    pairs = args.PAIRS
    solo = args.SOLO

    n_type = None
    if 'RNA' in fn or 'rna' in fn:
        n_type='RNA'
    else:
        n_type="DNA"

    #fn = "./ont_models/r9.4_180mv_450bps_6mer_DNA.model"
    #fn = "./ont_models/r9.4_180mv_70bps_5mer_RNA.model"
    if pairs:
        base_pairs = [("G","C"),("G", "A"), ("G","T"), ("C", "A"), ("C", "T"), ("A", "T")]
        
        manager = Manager()
        gpu_n = np.arange(get_available_gpus())
        avail_gpus = manager.list(gpu_n)
        res_dict = manager.dict()
        
        for base_pair in base_pairs:
            key = '-'.join(list(base_pair))
            res_dict[key] = manager.dict()
            res_dict[key]['r'] = manager.list()
            res_dict[key]['r2'] = manager.list()
            res_dict[key]['rmse'] = manager.list()
            res_dict[key]['train_history'] = manager.list()
            res_dict[key]['train_kmers'] = manager.list()
            res_dict[key]['test_kmers'] = manager.list()
            res_dict[key]['train_labels'] = manager.list()
            res_dict[key]['test_labels'] = manager.list()
            res_dict[key]['test_pred'] = manager.list()
            res_dict[key]['train_pred'] = manager.list()


        po = Pool(len(gpu_n))
        
        r = po.map_async(wrap ,
                         ((base_pair, avail_gpus, res_dict, fn, n_type) for base_pair in base_pairs))
        
        r.wait()
        print(r.get())
        po.close()
        po.join()

        new_res_dict = {}
        for base_pair in res_dict.keys():
            new_res_dict[base_pair] = {}

            for key in res_dict[base_pair].keys():
                new_res_dict[base_pair][key] = list(res_dict[base_pair][key])


        local_out = str(os.environ['MYOUT']) # see job.yml for env definition
        np.save('.'+local_out+out, new_res_dict) #this will go to /results/
    if solo:
        bases = ['A','T','C','G']
       
        manager = Manager()
        gpu_n = np.arange(get_available_gpus())
        avail_gpus = manager.list(gpu_n)
        res_dict = manager.dict()
        
        for base in bases:
            key = base
            res_dict[key] = manager.dict()
            res_dict[key]['r'] = manager.list()
            res_dict[key]['r2'] = manager.list()
            res_dict[key]['rmse'] = manager.list()
            res_dict[key]['train_history'] = manager.list()
            res_dict[key]['train_kmers'] = manager.list()
            res_dict[key]['test_kmers'] = manager.list()
            res_dict[key]['train_labels'] = manager.list()
            res_dict[key]['test_labels'] = manager.list()
            res_dict[key]['test_pred'] = manager.list()
            res_dict[key]['train_pred'] = manager.list()
        
        po = Pool(len(gpu_n))

        r = po.map_async(wrap ,
                         ((tuple((base)), avail_gpus, res_dict, fn, n_type) for base in bases))

        r.wait()
        print(r.get())
        po.close()
        po.join()

        new_res_dict = {}
        for base in res_dict.keys():
            new_res_dict[base] = {}

            for key in res_dict[base].keys():
                new_res_dict[base][key] = list(res_dict[base][key])


        local_out = str(os.environ['MYOUT']) # see job.yml for env definition
        np.save('.'+local_out+out, new_res_dict) #this will go to /results/

if __name__ == "__main__":
    main() 
