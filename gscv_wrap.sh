echo "running script..."
python gscv_main.py -i $1 -k $2 
echo "uploading files..."
#python3 s3upload.py $2
aws --endpoint $PRP s3 cp ./${MYOUT}/rna_best_params.npy s3://stuartlab/users/ianastop/out/
aws --endpoint $PRP s3 cp ./${MYOUT}/rna_cv_results.npy s3://stuartlab/users/ianastop/out/
echo "upload complete"
