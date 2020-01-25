import boto3
import sys
import os

prp = str(os.environ['PRP'])
local_out = str(os.environ['MYOUT']) # see job.yml for env definition
s3out = str(os.environ['S3OUT']) # see job.yml for env definition

local_out = './'
s3out = "users/ianastop/out/"

fn = str(sys.argv[1]) # file in loca_out, produced by script -- see wrap.sh

session = boto3.session.Session(profile_name="default")
bucket = session.resource("s3", endpoint_url=prp).Bucket("stuartlab")
bucket.upload_file(local_out+fn, s3out+fn) # will upload to S3OUT

