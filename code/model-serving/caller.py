import boto3
import json
client = boto3.client('lambda', region_name='eu-west-1', aws_access_key_id='AKIAYZYQQOACW4NY577K',
    aws_secret_access_key='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',)

res = client.invoke(
    FunctionName='arn:aws:lambda:eu-west-1:605088149509:function:mytestfunction',
    Payload=json.dumps(
       {
        "groups" : ["groups"], 
        "projections": ["count"],
        "filters": {"order_dow_lb": 4.0, "order_dow_ub": 4.0}

       }    
      ) 
    )
print(res)
print(json.loads(res['Payload'].read()))
