aws --endpoint $PRP s3 cp ${PRPOUT}rna_posdrop_repeat50times_run1_results.npy ./results/
aws --endpoint $PRP s3 cp ${PRPOUT}rna_posdrop_repeat50times_run2_results.npy ./results/

aws --endpoint $PRP s3 cp ${PRPOUT}rna_exclude_basepairs_run1_results.npy ./results/
aws --endpoint $PRP s3 cp ${PRPOUT}rna_exclude_basepairs_run2_results.npy ./results/

aws --endpoint $PRP s3 cp ${PRPOUT}rna_exclude_base_repeat50times_run1_results.npy ./results/
aws --endpoint $PRP s3 cp ${PRPOUT}rna_exclude_base_repeat50times_run2_results.npy ./results/

aws --endpoint $PRP s3 cp ${PRPOUT}rna_downsample_run1_results.npy ./results/
aws --endpoint $PRP s3 cp ${PRPOUT}rna_downsample_run2_results.npy ./results/

