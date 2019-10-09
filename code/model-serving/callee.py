import json
import boto3
import pickle


BUCKET_NAME = 'uogbigdata-models'
MODEL_FILE_NAME = 'model.pkl'

S3 = boto3.client('s3', region_name='eu-west-1')

def load_model(key):
    # Load model from S3 bucket
    response = S3.get_object(Bucket=BUCKET_NAME, Key=key)# Load pickle model
    model_str = response['Body'].read()
    model = pickle.loads(model_str)

    return model





def lambda_handler(event, context):
    # TODO implement
    data = event['data']

    # Load model
    model = load_model(MODEL_FILE_NAME)# Make prediction
    prediction = model.predict(data).tolist()# Respond with prediction result
    result = {'prediction': prediction}


    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }
