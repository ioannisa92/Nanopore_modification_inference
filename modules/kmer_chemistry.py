from rdkit import Chem
# from IPython.display import SVG
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
            #print(nAtms, m.shape[0], flush=True)
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

def get_AX(kmer_list, n_type="DNA", return_smiles=False):

    '''
    Function takes in a kmer-pA measurement lists. Kmers are converted to their SMILES representation.
    An array of the molecular Adjacency and Feature matrices is returned

    Parameters
    -----------
    kmer_list: list, list of  kmers

    Returns
    ----------
    A: mat, matrix of atom to atom connections for each kmer; shape = (n_of_molecules, n_atoms, n_atoms)
    X mat, matrix of atom to features for each kmer; shape = (n_of_molecules, n_atoms, n_features)
    '''

    k = len(kmer_list[0])

    dna_base = {"A": "OP(=O)(O)OCC1OC(N3C=NC2=C(N)N=CN=C23)CC1",
                "T": "OP(=O)(O)OCC1OC(N2C(=O)NC(=O)C(C)=C2)CC1",
                "G": "OP(=O)(O)OCC1OC(N2C=NC3=C2N=C(N)NC3=O)CC1",
                "C": "OP(=O)(O)OCC1OC(N2C(=O)N=C(N)C=C2)CC1",
                "M": "OP(=O)(O)OCC1OC(N2C(=O)N=C(N)C(C)=C2)CC1", # 5mC
                "Q": "OP(=O)(O)OCC1OC(N3C=NC2=C(NC)N=CN=C23)CC1", #6mA
                'K': "OP(=O)(O)OCC1OC(N2C(=O)N=C(N)C(CO)=C2)CC1"} # 5hmC 


    rna_base = {'A':'OP(=O)(O)OCC1OC(N3C=NC2=C(N)N=CN=C23)C(O)C1',
                'U':'OP(=O)(O)OCC1OC(N2C(=O)NC(=O)C=C2)C(O)C1',
                'G':'OP(=O)(O)OCC1OC(N2C=NC3=C2N=C(N)NC3=O)C(O)C1',
                'C':'OP(=O)(O)OCC1OC(N2C(=O)N=C(N)C=C2)C(O)C1',
                'X':'OP(=O)(O)OCC1OC(N3C=NC2=C(NC)N=CN=C23)C(O)C1', #6mA
                'Y':'OP(=O)(O)OCC1OC(N3C=NC2=C(N(C)C)N=CN=C23)C(O)C1', #6mmA
                'Z':'OP(=O)(O)OCC1OC(N2C=NC3=C2N=C(NC)NC3=O)C(O)C1', #2mG
                'Q':'OP(=O)(O)OCC1OC(C2C(=O)NC(=O)NC=2)C(O)C1', #pseudo-I
                'I':'OP(=O)(O)OCC1OC(N3C=NC2C(=O)NC=NC=23)C(O)C1'} # inosine 
    
    # rna smiles are in lower case
    dna_rna_base = {"A": "OP(=O)(O)OCC1OC(N3C=NC2=C(N)N=CN=C23)CC1",
                "T": "OP(=O)(O)OCC1OC(N2C(=O)NC(=O)C(C)=C2)CC1",
                "G": "OP(=O)(O)OCC1OC(N2C=NC3=C2N=C(N)NC3=O)CC1",
                "C": "OP(=O)(O)OCC1OC(N2C(=O)N=C(N)C=C2)CC1",
                "M": "OP(=O)(O)OCC1OC(N2C(=O)N=C(N)C(C)=C2)CC1", # 5mC
                "Q": "OP(=O)(O)OCC1OC(N3C=NC2=C(NC)N=CN=C23)CC1", #6mA
                'K': "OP(=O)(O)OCC1OC(N2C(=O)N=C(N)C(CO)=C2)CC1", # 5hmC 
                "a": "OP(=O)(O)OCC1OC(N3C=NC2=C(N)N=CN=C23)C(O)C1",
                "u": "OP(=O)(O)OCC1OC(N2C(=O)NC(=O)C=C2)C(O)C1", #U
                "g": "OP(=O)(O)OCC1OC(N2C=NC3=C2N=C(N)NC3=O)C(O)C1",
                "c": "OP(=O)(O)OCC1OC(N2C(=O)N=C(N)C=C2)C(O)C1",
                "q": "OP(=O)(O)OCC1OC(C2C(=O)NC(=O)NC=2)C(O)C1", #pseudo-I
                "i": "OP(=O)(O)OCC1OC(N3C=NC2C(=O)NC=NC=23)C(O)C1"} # inosine  




    if n_type=="DNA":
        smiles = get_kmer_smiles(k, dna_base)
        smiles = [smiles.get(kmer)[0] for kmer in kmer_list]
        
        A, X = get_AX_matrix(smiles, ['C', 'N', 'O', 'P'], 133)

    elif n_type=="RNA":
        smiles = get_kmer_smiles(k, rna_base)
        smiles = [smiles.get(kmer)[0] for kmer in kmer_list]

        A, X = get_AX_matrix(smiles, ['C', 'N', 'O', 'P'], 121)

    elif n_type == 'DNA_RNA':
        #smiles = []
        
        all_smiles = []
        for km in kmer_list:
            kmer_smiles = []
            for base in km:
                sm = dna_rna_base[base]
                kmer_smiles.append(sm)
            kmer_smiles = ''.join(kmer_smiles)+"O"
            all_smiles.append(kmer_smiles)  
        
        #for kmer in tqdm(kmer_list):
        #    k = len(kmer)
        #    smiles.append(get_kmer_smiles(k, dna_rna_base).get(kmer)[0])
        A, X = get_AX_matrix(all_smiles, ['C', 'N', 'O', 'P'],133) #padding everything to largest DNA kmer
            
    if return_smiles:
        return A,X,smiles
    return A,X


#functions
