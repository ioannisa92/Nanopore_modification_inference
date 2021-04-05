echo "running script..."
python combined_main.py -v $1 -o $2 -test_splits $3
echo "uploading files..."
#python3 s3upload.py $2
aws --endpoint $PRP s3 cp ./${MYOUT}/$2 s3://stuartlab/users/ianastop/out/
echo "upload complete"
