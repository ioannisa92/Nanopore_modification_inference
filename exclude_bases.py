#! /usr/bin/env python
import os
import numpy as np
from multiprocessing import Pool, Manager
import subprocess
from modules import kmer_chemistry
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import tqdm
from modules.cv_utils import *
import tensorflow as tf
from keras import backend as K
from modules.nn_model import *
from keras.callbacks import EarlyStopping

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

def rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_pred - y_true)))

def get_AX(kmer_list):

    '''
    Function takes in a kmer-pA measurement lists. Kmers are converted to their SMILES representation.
    An array of the molecular Adjacency and Feature matrices is returned

    Parameters
    -----------
    kmer_list: list, list of  kmerse

    Returns
    ----------
    A: mat, matrix of atom to atom connections for each kmer; shape = (n_of_molecules, n_atoms, n_atoms)
    X mat, matrix of atom to features for each kmer; shape = (n_of_molecules, n_atoms, n_features)
    '''

    k = len(kmer_list[0])

    dna_base = {'A':'OP(=O)(O)OCC1OC(N3C=NC2=C(N)N=CN=C23)CC1',
            'T':'OP(=O)(O)OCC1OC(N2C(=O)NC(=O)C(C)=C2)CC1',
            'G':'OP(=O)(O)OCC1OC(N2C=NC3=C2N=C(N)N=C3O)CC1',
            'C':'OP(=O)(O)OCC1OC(N2C(=O)N=C(N)C=C2)CC1',
            'M':'OP(=O)(O)OCC1OC(N2C(=O)N=C(N)C(C)=C2)CC1'}

    rna_base = {'A':'OP(=O)(O)OCC1OC(N3C=NC2=C(N)N=CN=C23)C(O)C1',
                'T':'OP(=O)(O)OCC1OC(N2C(=O)NC(=O)C=C2)C(O)C1',
                'G':'OP(=O)(O)OCC1OC(N2C=NC3=C2N=C(N)N=C3O)C(O)C1',
                'C':'OP(=O)(O)OCC1OC(N2C(=O)N=C(N)C=C2)C(O)C1',
                'Q':'OP(=O)(O)OCC1OC(C2C(=O)NC(=O)NC=2)C(O)C1'}


    if n_type=="DNA":
        dna_smiles = kmer_chemistry.get_kmer_smiles(k, dna_base)
        dna_smiles = [dna_smiles.get(kmer)[0] for kmer in kmer_list]

        A, X = kmer_chemistry.get_AX_matrix(dna_smiles, ['C', 'N', 'O', 'P'], 133)

    elif n_type=="RNA":
        rna_smiles = kmer_chemistry.get_kmer_smiles(k, rna_base)
        rna_smiles = [rna_smiles.get(kmer)[0] for kmer in kmer_list]

        A, X = kmer_chemistry.get_AX_matrix(rna_smiles, ['C', 'N', 'O', 'P'], 116)
    return A,X 
    
def fold_training(kmer_train,kmer_test, pA_train,pA_test, val_split=0,callbacks=True):

    '''
    Function takes in train and test matrices and fits a new randomly initialized model.
    Function records training loss during training, validation (if selected), and also
    calculates pearson correlation, R2, and RMSE score on the test set

    Parameters
    ----------
    kmer_train: numpy mat; training matrix of kmers
    kmer_tedt: numpy mat; test matrix of kmers
    pA_train: numpy mat; training taget values of pA for kmers in kmer_train set
    pA_test: numpy mat; test target values of pA for kmers in kmer_test set
    val_split: int; percent of data to use as validation set during training
    callbacks: bool; whether to use callbacks in training

    Returns
    --------
    train_hist: dict; dictionary of training loss or validation loss if selected
    r: float; pearson correlation of predicted v. target values
    r2: float; R2 correlation of predicted v. target values
    rmse_score: float; RMSE score correlation of predicted v. target values

    '''
    A_train, X_train = get_AX(kmer_train)
    gcn_filters_train = initialize_filters(A_train)
    A_test, X_test = get_AX(kmer_test)
    gcn_filters_test = initialize_filters(A_test)
    
 
    # initializing model - new randomly initialized model for every fold training
    if n_type=="DNA":
        model = initialize_model(X_train, gcn_filters_train, n_gcn=4, n_cnn=4, kernal_size_cnn=4, n_dense=4)
    elif n_type=="RNA":
        model = initialize_model(X_train, gcn_filters_train, n_gcn=1, n_cnn=4, kernal_size_cnn=4, n_dense=4)
    model.compile(loss='mean_squared_error', optimizer=Adam())

    if callbacks:
        callbacks = [EarlyStopping(monitor='loss', min_delta=0.01, patience=10, verbose=1, mode='min', baseline=None, restore_best_weights=False)]
    else:
        callbacks = None

    # training model and testing performance
    train_hist = model.fit([X_train,gcn_filters_train],pA_train,validation_split=val_split, batch_size=128, epochs=350, verbose=0, callbacks=callbacks)
    test_pred = model.predict([X_test, gcn_filters_test]).flatten()

    #calculating metrics
    r, _ = pearsonr(pA_test, test_pred)
    r2 = r2_score(pA_test, test_pred)
    rmse_score = rmse(test_pred, pA_test)

    # clearing session to avoid adding unwanted nodes on TF graph
    K.clear_session()
    return train_hist, r, r2, rmse_score, test_pred

def main():
    global fn
    fn = "./ont_models/r9.4_180mv_450bps_6mer_DNA.model"

    global n_type
    n_type = None
    if 'RNA' in fn or 'rna' in fn:
        n_type='RNA'
    else:
        n_type="DNA"

    local_out = str(os.environ['MYOUT']) # see job.yml for env definition
 
    base_pairs = [("G","C"),("G", "A"), ("G","T"), ("C", "A"), ("C", "T"), ("A", "T")]
    
    res={} 
     
    for base_pair in base_pairs:
        print("running", base_pair)
        key = '-'.join(list(base_pair))
        res[key] = {'r':[], 'r2':[],'rmse':[], "train_history":[],'train_kmers':[],'test_kmers':[], 'train_labels':[], 'test_labels':[], 'test_pred' : []}
        
        c = GetTrainTest(fn, base_pair)
        train_kmers, train_pa = c.get_train()
        test_kmers, test_pa = c.get_test()
        
        
        #train model
        train_hist, r, r2, rmse_score, test_pred = fold_training(train_kmers,
                                                                    test_kmers,
                                                                    train_pa,
                                                                    test_pa,
                                                                    val_split=0,
                                                                    callbacks=True)
        res[key]['r'] += [r]
        res[key]['r2'] += [r2]
        res[key]['rmse'] += [rmse_score]

        res[key]['train_history']  += [train_hist.history]
        res[key]['train_kmers'] += [train_kmers]
        res[key]['test_kmers'] += [test_kmers]
        res[key]['train_labels'] += [train_pa]
        res[key]['test_labels'] += [test_pa]
        res[key]['test_pred'] += [test_pred]

    np.save('.'+local_out+"exclude_base_results.npy", res) #this will go to /results/
if __name__ == "__main__":
    main() 
