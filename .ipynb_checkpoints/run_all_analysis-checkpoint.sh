
        
#command: ["bash", "gscv_wrap.sh", "./ont_models/r9.4_GSE124309_atgc_xtgc.model", "10"]
#command: ["bash", "gscv_wrap.sh", "./ont_models/r9.4_180mv_450bps_6mer_DNA.model", "10"]
#command: ["bash", "gscv_wrap.sh", "./ont_models/6mer_DNA_5mer_RNA_combined.model", "10"]
        
#command: ["bash", "basepair_exclude_wrap.sh", "./ont_models/r9.4_180mv_450bps_6mer_DNA.model", "dna_exclude_basepairs_run2_results.npy", 'DNA']
#command: ["bash", "basepair_exclude_wrap.sh", "./ont_models/r9.4_180mv_70bps_5mer_5to3_RNA.model", "rna_exclude_basepairs_run2_results.npy", 'RNA']

#command: ["bash", "base_exclude_wrap.sh", "./ont_models/r9.4_180mv_450bps_6mer_DNA.model", "dna_exclude_base_repeat50times_run4_results.npy", 'DNA']
#command: ["bash", "base_exclude_wrap.sh", "./ont_models/r9.4_180mv_70bps_5mer_5to3_RNA.model", "rna_exclude_base_repeat50times_run1_results.npy", 'RNA']
#command: ["bash", "base_exclude_wrap.sh", "./ont_models/6mer_DNA_5mer_RNA_combined.model", "dna_rna_exclude_base_repeat50times_run1_results.npy"]

#command: ["bash", "cv_wrap.sh", "./ont_models/r9.4_180mv_450bps_6mer_DNA.model","dna_downsample_run2_results.npy"]
#command: ["bash", "cv_wrap.sh", "./ont_models/r9.4_180mv_70bps_5mer_5to3_RNA.model","rna_downsample_run2_results.npy" ]
#command: ["bash", "cv_wrap.sh", "./ont_models/6mer_DNA_5mer_RNA_combined.model","dna_rna_cv_results_run2.npy" ]

#command: ["bash", "combined_cv_wrap.sh", "0", "combined_cv_results.npy",  "0.9 0.75 0.5 0.25 0.1"]

#command: ["bash", "kmer_cv_wrap.sh", "./ont_models/r9.4_180mv_70bps_5mer_5to3_RNA.model","rna_posdrop_repeat50times_run2_results.npy" ]
#command: ["bash", "kmer_cv_wrap.sh", "./ont_models/r9.4_180mv_450bps_6mer_DNA.model","dna_posdrop_repeat50times_run2_results.npy" ]


command: ['bash','rna_mod_pred.sh', './ont_models/r9.4_GSE124309_atgc_xtgc.model', 'rna_model','rna_mod_pred_results_repeat50times']
'bash','dna_mod_trainpred.sh', './ont_models/r9.4_180mv_450bps_6mer_DNA.model', 'dna_model','dna_mod_trainpred_results_50fold_run2']


['bash','dna_mod_pred.sh', './ont_models/r9.4_180mv_450bps_6mer_DNA.model','dna_model', 'dna_mod_pred_results_50repeat_run1.npy']