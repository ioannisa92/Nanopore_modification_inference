#! /usr/bin/env python
from sklearn.model_selection import ShuffleSplit
import numpy as np
'''
Script will generate cross validation folds in two ways:
1 - Random 90-10, 80-20, ... splits to test how little information we need to
    train on.
2 - CV splits with missing bases in the training, which are present in the test
    set
'''



def kmer_parser(fn):
    '''
    Function parses kmer file and returns

    Parameters
    ----------
    fn: str, path to file

    Returns
    ----------
    kmer_list: array, list of kmers in the order they appear in the file
    pA_list: array, list of pA values (floats) in the same order
    '''
    kmer_list = []
    pA_list = []
    with open(fn ,'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split('\t')
            kmer = str(line[0]).strip()
            pA = float(line[1].strip())
            kmer_list += [kmer]
            pA_list += [pA]

    return np.array(kmer_list), np.array(pA_list)

def cv_folds(X, Y, folds=5):
    '''
    Function takes in a kmer_list and returns cv fold indeces.
    An array of test_sizes ranging from 0.1-0.9 is made
    For each test size a train/test split is make for kmers and their pAs

    Parameters
    -----------
    X: array, list of kmers
    Y: array, list of the pA of those kmers
    folds, int, number of CV folds to be made

    Returns
    -----------
    kmer_train_mat: mat, shape(folds, train_size) for each train/test split
    kmer_test_mat: mat, shape(folds, train_size) for each train/test split

    pA_train_mat: mat, shape(folds, test_size) for each train/test split
    pA_test_mat: mat, shape(folds, test_size) for each train/test split
    '''

    #test_sizes = np.arange(0,1,0.1)[1:] #excluding zero
    test_sizes = [0.75, 0.5, 0.25]

    for test_size in test_sizes:

        # The following matrices contain the train/test kmer and corresponding pA values
        # for the folds produced. Shape is (folds, train/test size)
        kmer_train_mat = []
        kmer_test_mat = []

        pA_train_mat = []
        pA_test_mat = []

        splitter = ShuffleSplit(n_splits=folds, test_size=test_size, random_state=42).split(X)

        for train_idx, test_idx in (splitter):

            x_train = X[train_idx]
            x_test = X[test_idx]

            y_train = Y[train_idx]
            y_test = Y[test_idx]

            kmer_train_mat += [x_train]
            kmer_test_mat += [x_test]

            pA_train_mat +=[y_train]
            pA_test_mat += [y_test]

        kmer_train_mat = np.vstack(kmer_train_mat)
        kmer_test_mat = np.vstack(kmer_test_mat)

        pA_train_mat = np.vstack(pA_train_mat)
        pA_test_mat = np.vstack(pA_test_mat)

        yield test_size,kmer_train_mat,kmer_test_mat,pA_train_mat,pA_test_mat


def base_folds(kmer_list, pA_list):

    '''
    Function generates train test splits based on DNA base position on the kmer
    For example:
        A train split will contain no A bases in the first position.
        The test split will contain all kmers with A in the first position
        And so on...

    Each train/test matrix will have shape(n_bases[4], train/test size)
    The order of the bases, meaning rows, in these matrices is [A,T,C,G].
    Six matrices will be produced with size (4,train/test size)

    For the train matrices, the base is ABSENT
    For the test matrices, the base in PRSENT
    '''

    bases = ['A', 'T', 'C', 'G']
    positions = np.arange(0,len(kmer_list[0])) # assuming all kmers in kmer_list are of equal length

    new_kmer_list = []
    for kmer in kmer_list:
        new_kmer_list += [list(kmer)]
    new_kmer_list = np.vstack(new_kmer_list)

    for pos in positions:
        print("examining position:",pos)
        pos_bases = new_kmer_list[:,pos]

        kmer_train_mat = []
        kmer_test_mat = []

        pA_train_mat = []
        pA_test_mat = []

        for base in bases:
            # print("examining base:",base)
            train_idx = np.argwhere(pos_bases!=base).flatten() #base in this pos absent from training
            test_idx = np.argwhere(pos_bases==base).flatten() #base in this pos present in testing

            kmer_train_mat += [kmer_list[train_idx]]
            kmer_test_mat += [kmer_list[test_idx]]

            pA_train_mat += [pA_list[train_idx]]
            pA_test_mat += [pA_list[test_idx]]

            # print(np.argwhere(pos_bases==base).flatten().shape) #test
            # print(np.argwhere(pos_bases!=base).flatten().shape) #train

        kmer_train_mat = np.vstack(kmer_train_mat)
        kmer_test_mat = np.vstack(kmer_test_mat)

        pA_train_mat = np.vstack(pA_train_mat)
        pA_test_mat = np.vstack(pA_test_mat)

        # print(kmer_train_mat.shape)
        # print(kmer_test_mat.shape)
        #
        # print(pA_train_mat.shape)
        # print(pA_test_mat.shape)

        yield pos, kmer_train_mat,kmer_test_mat,pA_train_mat,pA_test_mat



if __name__ == "__main__":

    fn = "../ont_models/r9.4_180mv_450bps_6mer_DNA.model"

    kmer_list, pA_list = kmer_parser(fn)
    for kmer_train_mat,kmer_test_mat,pA_train_mat,pA_test_mat in cv_folds(kmer_list, pA_list):
        print(kmer_train_mat.shape)
        print(kmer_test_mat.shape)
        print(pA_train_mat.shape)
        print(pA_test_mat.shape)

    for kmer_train_mat,kmer_test_mat,pA_train_mat,pA_test_mat in base_folds(kmer_list, pA_list):
        print(kmer_train_mat.shape)
        print(kmer_test_mat.shape)
        print(pA_train_mat.shape)
        print(pA_test_mat.shape)

        print(kmer_train_mat[0])
        print(kmer_test_mat[0])
