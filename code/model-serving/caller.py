import boto3
import json
client = boto3.client('lambda', region_name='eu-west-1', aws_access_key_id='AKIAYZYQQOACW4NY577K',
    aws_secret_access_key='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',)

res = client.invoke(
    FunctionName='arn:aws:lambda:eu-west-1:605088149509:function:mytestfunction',
    Payload=json.dumps({
        "data":[[6.2, 3.4], [6.2, 1]]
        })
    )
print(res)
print(json.loads(res['Payload'].read()))
