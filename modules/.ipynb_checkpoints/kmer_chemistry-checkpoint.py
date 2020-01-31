#! /usr/bin/env python

from rdkit import Chem
#from IPython.display import SVG
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from collections import defaultdict
import numpy as np
import itertools
#imports


def get_kmer_smiles(k, base):
    '''
    get SMILES string for kmer
    '''
    bs = []
    sm = []
    for b,s in base.items():
        bs.append(b)
        sm.append(s)
    bs = list(itertools.product(*([bs] * k)))
    sm = list(itertools.product(*([sm] * k)))
    
    smiles = defaultdict(list)
    for i in list(range(0, len(bs))):
        smiles[''.join(bs[i])].append(''.join(sm[i])+'O')
    
    return smiles
    
    
def plot_molecule(smiles, size=(600, 300), kekulize=True):
    '''
    visualize a single SMILES string
    code adapted from https://www.kaggle.com/corochann/visualize-molecules-with-rdkit
    '''
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(mol)
        except:
            mol = Chem.Mol(mol.ToBinary())
    if not mol.GetNumConformers():
        rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return SVG(svg.replace('svg:',''))


def get_n_hydro(smiles):
    '''
    get number of Hs
    '''
    mol = Chem.MolFromSmiles(smiles)
    before = mol.GetNumAtoms()
    mol = Chem.AddHs(mol)
    after = mol.GetNumAtoms()
    nH = after - before
    return nH


def get_compound_graph(smiles, Atms):
    '''
    we follow the pipeline developed by Duvenaud et al. Convolutional networks on graphs for learning molecular fingerprints, Advances in neural information processing systems, 2015; pp 2224-2232
    function returns adjacency (A) and feature matrix (X)
    '''  
    mol = Chem.MolFromSmiles(smiles)    
    X = np.zeros((mol.GetNumAtoms(), len(Atms) + 4))
    #feature matrix [unique atoms, atom_degree, nH, implicit valence, aromaticity indicator] 
    A = np.zeros((mol.GetNumAtoms(), mol.GetNumAtoms()))
    #adjency matrix (binary) indicating which atom is connected to each other atom
    
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        symbol = mol.GetAtomWithIdx(atom_idx).GetSymbol()
        symbol_idx = Atms.index(symbol)
        atom_degree = len(atom.GetBonds())
        implicit_valence = mol.GetAtomWithIdx(atom_idx).GetImplicitValence()
        X[atom_idx, symbol_idx] = 1
        X[atom_idx, len(Atms)] = atom_degree
        X[atom_idx, len(Atms) + 1] = get_n_hydro(symbol)
        X[atom_idx, len(Atms) + 2] = implicit_valence
        if mol.GetAtomWithIdx(atom_idx).GetIsAromatic():
                X[atom_idx, len(Atms)+3] = 1
        
        for n in (atom.GetNeighbors()):
            neighbor_atom_idx = n.GetIdx()
            A[atom_idx, neighbor_atom_idx] = 1
    
    return A, X


def pad_compound_graph(mat_list, nAtms, axis=None):
        '''
        MutliGraphCNN assumes that the number of nodes for each graph in the dataset is same.
        for graph with arbitrary size, we append all-0 rows/columns in adjacency and feature matrices and based on max graph size
        function takes in a list of matrices, and pads them to the max graph size
        assumption is that all matrices in there should be symmetric (#atoms x #atoms)
        output is a concatenated version of the padded matrices from the lsit

        '''
        assert type(mat_list) is list
        padded_matrices = []
        for m in mat_list:
            pad_length = nAtms - m.shape[0]
            if axis==0:
                padded_matrices += [np.pad(m, [(0,pad_length),(0,0)], mode='constant')]
            elif axis is None:
                padded_matrices += [np.pad(m, (0,pad_length), mode='constant')]
        
        return np.vstack(padded_matrices)


def get_AX_matrix(smiles, Atms, nAtms):
        '''
        get A and X matrices from a list of SMILES strings
        '''
        A_mat_list = []
        X_mat_list = []
        for sm in smiles:
            A, X = get_compound_graph(sm, Atms)
            A_mat_list += [A]
            X_mat_list += [X]
        padded_A_mat = pad_compound_graph(A_mat_list, nAtms)
        padded_X_mat = pad_compound_graph(X_mat_list, nAtms, axis=0)
        
        padded_A_mat = np.split(padded_A_mat, len(smiles), axis=0)
        padded_A_mat = np.array(padded_A_mat)
        padded_X_mat = np.split(padded_X_mat, len(smiles), axis=0)
        padded_X_mat = np.array(padded_X_mat)

        return padded_A_mat, padded_X_mat  
#functions
