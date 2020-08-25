echo "running script..."
python cv_main.py -i $1 -cv -o $2 
echo "uploading files..."
#python3 s3upload.py $2
aws --endpoint $PRP s3 cp ./${MYOUT}/$2 s3://stuartlab/users/ianastop/out/
echo "upload complete"
