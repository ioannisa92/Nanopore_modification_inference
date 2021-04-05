python gscv_main.py -i ./ont_models/r9.4_180mv_70bps_5mer_5to3_RNA.model -k 10
python gscv_main.py -i ./ont_models/r9.4_180mv_450bps_6mer_DNA.model -k 10

python cv_main.py -i ./ont_models/r9.4_180mv_450bps_6mer_DNA.model -cv -o dna_downsample_results.npy
python cv_main.py -i ./ont_models/r9.4_180mv_70bps_5mer_5to3_RNA.model -cv -o rna_downsample_results.npy

python cv_main.py -i ./ont_models/r9.4_180mv_70bps_5mer_5to3_RNA.model -kmer_cv -o rna_posdrop_results.npy
python cv_main.py -i ./ont_models/r9.4_180mv_450bps_6mer_DNA.model -kmer_cv -o dna_posdrop_results.npy

python exclude_bases.py -i ./ont_models/r9.4_180mv_450bps_6mer_DNA.model -o dna_exclude_base_results.npy -n_type $3 'DNA' -base_exclude
python exclude_bases.py -i ./ont_models/r9.4_180mv_70bps_5mer_5to3_RNA.model -o rna_exclude_base_results.npy -n_type $3 'RNA' -base_exclude

python exclude_bases.py -i ./ont_models/r9.4_180mv_450bps_6mer_DNA.model -o dna_exclude_basepairs_results.npy -n_type $3 'DNA' -base_pair_exclude
python exclude_bases.py -i ./ont_models/r9.4_180mv_70bps_5mer_5to3_RNA.model -o rna_exclude_basepairs_results.npy -n_type $3 'RNA' -base_pair_exclude

# de novo modification prediction
python dna_mod_pred.py -i  ./ont_models/r9.4_180mv_450bps_6mer_DNA.model -model_fn dna_model -o dna_mod_pred_50repeat_results.npy

# de novo and fractions of modified kmers included in training
python dna_mod_trainpred.py -i ./ont_models/r9.4_180mv_450bps_6mer_DNA.model  -o dna_mod_trainpred_results_50fold.npy







