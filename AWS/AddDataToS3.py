import boto3
import sys


def uploadToS3(bucket, filename):
    s3 = boto3.resource('s3')
    s3.Bucket(bucket).put_object(Key=filename)


bucket = sys.argv[1]
filename = sys.argv[2]

uploadToS3(bucket, filename)

