import boto3
import sys 

client = boto3.client('machinelearning')

def create_evaluation(dataSourceId):
	response = client.create_evaluation(
	    EvaluationId='SpotpriceEvaluation',
	    EvaluationName='SpotpriceEvaluation',
	    MLModelId='SpotpriceModel',
	    EvaluationDataSourceId=dataSourceId
	)


dataSourceId = sys.argv[1]

create_evaluation(dataSourceId)