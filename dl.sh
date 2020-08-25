
for i in $(seq 0 49);do
    #aws --endpoint $PRP s3 rm s3://stuartlab/users/ianastop/out/dna_model_repeat$i.h5 
    aws --endpoint $PRP s3 cp ${PRPOUT}dna_model_repeat$i.h5 ./saved_models
done

#aws --endpoint $PRP s3 cp s3://stuartlab/users/ianastop/out/dna_cv_results_run1.npy ./results
#aws --endpoint $PRP s3 cp s3://stuartlab/users/ianastop/out/dna_cv_results_run2.npy ./results
#aws --endpoint $PRP s3 cp s3://stuartlab/users/ianastop/out/dna_exclude_base_repeat50times_results.npy ./results
#aws --endpoint $PRP s3 cp s3://stuartlab/users/ianastop/out/dna_exclude_base_repeat50times_results_run2.npy ./results
#aws --endpoint $PRP s3 cp s3://stuartlab/users/ianastop/out/dna_exclude_basepairs_results_run1.npy ./results
#aws --endpoint $PRP s3 cp s3://stuartlab/users/ianastop/out/dna_exclude_basepairs_results_run2.npy ./results
#aws --endpoint $PRP s3 cp s3://stuartlab/users/ianastop/out/dna_kmer_cv_repeat50times_results_run1.npy ./results
#aws --endpoint $PRP s3 cp s3://stuartlab/users/ianastop/out/dna_kmer_cv_repeat50times_results_run2.npy ./results 
#aws --endpoint $PRP s3 cp s3://stuartlab/users/ianastop/out/dna_model_results_repeat50times.npy ./results

#aws --endpoint $PRP s3 cp s3://stuartlab/users/ianastop/out/rna_cv_results_run1.npy ./results
#aws --endpoint $PRP s3 cp s3://stuartlab/users/ianastop/out/rna_cv_results_run2.npy ./results
#aws --endpoint $PRP s3 cp s3://stuartlab/users/ianastop/out/rna_exclude_base_results_run1.npy ./results
#aws --endpoint $PRP s3 cp s3://stuartlab/users/ianastop/out/rna_exclude_base_results_run2.npy ./results
#aws --endpoint $PRP s3 cp s3://stuartlab/users/ianastop/out/rna_exclude_basepairs_results_run1.npy ./results
#aws --endpoint $PRP s3 cp s3://stuartlab/users/ianastop/out/rna_exclude_basepairs_results_run2.npy ./results
#aws --endpoint $PRP s3 cp s3://stuartlab/users/ianastop/out/rna_kmer_cv_repeat50times_results_run1.npy ./results
#aws --endpoint $PRP s3 cp s3://stuartlab/users/ianastop/out/rna_kmer_cv_repeat50times_results_run2.npy ./results
