import boto3
import sys 

client = boto3.client('machinelearning')

def create_ml_model(dataSourceId):
	response = client.create_ml_model(
	    MLModelId='SpotpriceModel',
	    MLModelName='Spotprice',
	    MLModelType='REGRESSION',
	    TrainingDataSourceId=dataSourceId
	)

dataSourceId = sys.argv[1]

create_ml_model(dataSourceId)