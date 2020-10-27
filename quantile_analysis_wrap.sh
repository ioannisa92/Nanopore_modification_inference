echo "downloading files..."

set -e
while read line
do
  aws --endpoint $PRP s3 cp s3://stuartlab/users/ianastop/out/$line /root/results/
done < quantile_analysis_filename.txt


echo "running script..."
python quantile_analysis.py -i $1 -model_fn $2 -o $3 -quantiles $4 -gscv_res $5
echo "uploading files..."
#python3 s3upload.py 2


aws --endpoint $PRP s3 cp ./${MYOUT}/$3.npy s3://stuartlab/users/ianastop/out/
echo "upload complete"
