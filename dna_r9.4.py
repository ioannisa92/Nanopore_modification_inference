import os
os.chdir('/Users/Ding/Desktop/de_novo_modification_inference/')
#working directory

from modules import kmer_chemistry
from modules import nn_model
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
#imports

base = {'A':'OP(=O)(O)OCC1OC(N3C=NC2=C(N)N=CN=C23)CC1',
        'T':'OP(=O)(O)OCC1OC(N2C(=O)NC(=O)C(C)=C2)CC1',
        'G':'OP(=O)(O)OCC1OC(N2C=NC3=C2N=C(N)N=C3O)CC1',
        'C':'OP(=O)(O)OCC1OC(N2C(=O)N=C(N)C=C2)CC1',
        'M':'OP(=O)(O)OCC1OC(N2C(=O)N=C(N)C(C)=C2)CC1'}
#modifications
#M, 5mC, OP(=O)(O)OCC1OC(N2C(=O)N=C(N)C(C)=C2)CC1
#O, 5hmC, OP(=O)(O)OCC1OC(N2C(=O)N=C(N)C(C)=C2)CC1
#E, EdU, OP(=O)(O)OCC1OC(N2C(=O)NC(=O)C(C#C)=C2)CC1
dna_smiles = kmer_chemistry.get_kmer_smiles(6, base)
kmer_chemistry.plot_molecule(dna_smiles.get('ATGCMA')[0])
#DNA SMILES strings

f = open('./ont_models/r9.4_180mv_450bps_6mer_DNA.model')
native = defaultdict(list)
for line in f:
    if ('#' in line) == False:
        native[line.split()[0]].append(float(line.split()[1]))
f.close() 
#native model 

smiles = [dna_smiles.get(key)[0] for key in native.keys()]
pas = [native.get(key)[0] for key in native.keys()]
A, X = kmer_chemistry.get_AX_matrix(smiles, ['C', 'N', 'O', 'P'], 133)
#represent native kmers

history = nn_model.tune_model(A, X, pas, [1, 2, 3, 4], [1, 2, 3, 4], [4, 8, 16], [1, 2, 3, 4], [0.2], validation_split=0.1, epochs=500)
nn_model.save_tuning_history('dna_native_r9.4', history)
nn_model.plot_epoch_loss(history, list(history.keys()))
nn_model.plot_epoch_loss(history, ['4_4_4_4_0.2'])
#tune model hyper-parameters

filters = nn_model.initialize_filters(A)
model = nn_model.initialize_model(X, filters, 4, 4, 4, 4, 0.2)
model = nn_model.train_model(X, pas, model, filters, validation_split=0.1, epochs=500)
nn_model.save_model('dna_native_r9.4', model)
#fit model

predict = nn_model.fit_model(X, model, filters)
fig = plt.gcf()
fig.set_size_inches(5, 5)
plt.scatter(pas, predict, color='black', s=2, alpha=0.5)
plt.plot(np.linspace(50,120,100), np.linspace(50,120,100), color='red')
plt.title('native kmers', fontsize=25)
plt.xlabel('model pA', fontsize=25)
plt.ylabel('predicted pA', fontsize=25)
plt.show()
#apply back to native kmers
