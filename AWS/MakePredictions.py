import boto3
import sys 

client = boto3.client('machinelearning')

def create_batch_prediction(dataSourceId):
	client = boto3.client('machinelearning')
	response = client.create_batch_prediction(
	    BatchPredictionId='spotPriceBatchPrediction',
	    BatchPredictionName='spotPriceBatchPrediction',
	    MLModelId='001',
	    BatchPredictionDataSourceId=dataSourceId,
	    OutputUri='s3://spotpricelyckander/predictions'
	)


dataSourceId = sys.argv[1]

create_batch_prediction(dataSourceId)