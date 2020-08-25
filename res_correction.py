import numpy as np
from modules.cv_utils import  kmer_parser

res = np.load('./results/dna_mod_pred_results_50fold.npy', allow_pickle=True).item()

kmer_file = './ont_models/r9.4_180mv_450bps_6mer_DNA.model'
kmer_list, pA_list = kmer_parser(kmer_file)

for train_split in res.keys():
    
    new_kmer_list = []
    for train_mod_kmer_list in res[train_split]['train_kmers']:
        new_kmer_list += [np.concatenate([kmer_list, train_mod_kmer_list], axis=0)]
    res[train_split]['train_kmers'] = new_kmer_list

np.save('./results/dna_mod_pred_results_50fold_corrected.npy', res)
