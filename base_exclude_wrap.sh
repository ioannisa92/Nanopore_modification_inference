echo "running script..."
python exclude_bases_multi.py -i $1 -o $2 
echo "uploading files..."
#python3 s3upload.py $2
aws --endpoint $PRP s3 cp ./${MYOUT}/$2 s3://stuartlab/users/ianastop/out/
echo "upload complete"
