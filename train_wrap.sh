echo "running script..."
python train.py -i $1 -model_fn $2 -o $3
echo "uploading files..."
#python3 s3upload.py 2


aws --endpoint $PRP s3 cp ./${MYOUT}/$3.npy s3://stuartlab/users/ianastop/out/
echo "upload complete"
