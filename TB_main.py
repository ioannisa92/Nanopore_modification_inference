#! /usr/bin/env python
from modules.kmer_cv import *
from modules import kmer_chemistry
from modules.nn_model import *
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
#imports


'''
Script trains on all data of the 9.4 dna model and test on all data.
This will the theoretical best performance the model can achieve given overfitting.

'''

def rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_pred - y_true)))

base = {'A':'OP(=O)(O)OCC1OC(N3C=NC2=C(N)N=CN=C23)CC1',
        'T':'OP(=O)(O)OCC1OC(N2C(=O)NC(=O)C(C)=C2)CC1',
        'G':'OP(=O)(O)OCC1OC(N2C=NC3=C2N=C(N)N=C3O)CC1',
        'C':'OP(=O)(O)OCC1OC(N2C(=O)N=C(N)C=C2)CC1',
        'M':'OP(=O)(O)OCC1OC(N2C(=O)N=C(N)C(C)=C2)CC1'}

dna_smiles = kmer_chemistry.get_kmer_smiles(6, base)

fn = "./ont_models/r9.4_180mv_450bps_6mer_DNA.model"
kmer_list, pA_list = kmer_parser(fn)

smiles = [dna_smiles.get(kmer)[0] for kmer in kmer_list]
A, X = kmer_chemistry.get_AX_matrix(smiles, ['C', 'N', 'O', 'P'], 133)


#fitting model with all the data
filters = initialize_filters(A)
model = initialize_model(X, filters, 4, 4, 4, 4, 0.2)
#training history 
results, model = train_model(X, pA_list, model, filters, validation_split=0.1, epochs=350)

#predicting on all data
pA_pred = model.predict([X, filters])
pred_rmse = rmse(pA_list, pA_pred)

results["pred_rmse"] = pred_rmse
np.save("./results/theoretical_best.npy", results)
