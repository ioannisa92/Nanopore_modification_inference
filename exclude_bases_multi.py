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

        self.kmer_mat1, _ = kmer_parser(self.fn, exclude_base=self.base_pair[0])
        self.kmer_mat2, _ = kmer_parser(self.fn, exclude_base=self.base_pair[1]) 
    
    def get_train(self):
        self.train_kmers = set(self.kmer_mat1)|set(self.kmer_mat2)
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
    base_pair = args[0]
    key = '-'.join(list(base_pair))
    avail_gpus = args[1]
    res_dict = args[2]
    fn = args[3]
    n_type=args[4]
    
    c = GetTrainTest(fn, base_pair)
    train_kmers, train_pa = c.get_train()
    test_kmers, test_pa = c.get_test()
    
 
    r,r2,rmse_score, train_hist, kmer_train, kmer_test, pA_train, pA_test, test_pred = run_model((train_kmers, train_pa, test_kmers, test_pa, n_type, avail_gpus))

    res_dict[key]['r'] += [r]
    res_dict[key]['r2'] += [r2]
    res_dict[key]['rmse'] += [rmse_score]

    res_dict[key]['train_history']  += [train_hist]
    res_dict[key]['train_kmers'] += [kmer_train]
    res_dict[key]['test_kmers'] += [kmer_test]
    res_dict[key]['train_labels'] += [pA_train]
    res_dict[key]['test_labels'] += [pA_test]
    res_dict[key]['test_pred'] += [test_pred]
    
    print("dict updated")

def main():
    ########----------------------Command line arguments--------------------##########
    parser = argparse.ArgumentParser(description="Run base-pair specific dropout cross validation")
    parser.add_argument('-i', '--FILE', default=None, type=str, required=False, help='kmer file with pA measurement')
    parser.add_argument('-o', '--OUT', default="out.npy", type=str, required=False, help='Full path for .npy file where results are saved')
    args=parser.parse_args()
    ########----------------------Command line arguments--------------------##########
    fn = args.FILE
    out = args.OUT

    n_type = None
    if 'RNA' in fn or 'rna' in fn:
        n_type='RNA'
    else:
        n_type="DNA"

    #fn = "./ont_models/r9.4_180mv_450bps_6mer_DNA.model"
    #fn = "./ont_models/r9.4_180mv_70bps_5mer_RNA.model"
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


    po = Pool(len(gpu_n))
    
    r = po.map_async(wrap ,
                     ((base_pair, avail_gpus, res_dict, fn, n_type) for base_pair in base_pairs))
    
    r.wait()
    print(r.get())
    po.close()
    po.join()

    new_res_dict = {}
    for drug in res_dict.keys():
        new_res_dict[drug] = {}

        for key in res_dict[drug].keys():
            new_res_dict[drug][key] = list(res_dict[drug][key])


    local_out = str(os.environ['MYOUT']) # see job.yml for env definition
    np.save('.'+local_out+out, new_res_dict) #this will go to /results/

if __name__ == "__main__":
    main() 
