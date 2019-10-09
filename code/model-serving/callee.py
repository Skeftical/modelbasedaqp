import json
# import boto3
import pickle
from sklearn.externals import joblib

with open('distinct_attribute_catalogue.pkl', 'rb') as f:
    distinct_attr = pickle.load(f)

with open('model_catalogue.pkl', 'rb') as f:
    model_catalogue = pickle.load(f)

with open('labels_catalogue.pkl', 'rb') as f:
    labels_catalogue = pickle.load(f)





def lambda_handler(event, context):
    # TODO implement
    proj_dict = event['projections']
    groups = event['groups']
    filters = event['filters']

    for p in proj_dict:
        res[p] = []
        est = model_catalogue[p]
        if len(gattr)>0:
            for g in groups:
                gvalues = distinct_attr[g]
                print("length of groupby values {}".format(len(gvalues)))
                res[g] = gvalues
                filters[g+'_lb'] = [labels_catalogue.get(gval,np.nan) for gval in gvalues]
                res[p]+=est.predict_many(filters).tolist()
        else:
            res[p].append(est.predict_one(filters))

    result = {'result': res}


    return {
        'statusCode': 200,
        'body': json.dumps(res)
    }


if __name__=="__main__":
    event = {
      "data": [
        [
          6.2,
          3.4
        ],
        [
          6.2,
          1
        ]
      ]
    }
    res = lambda_handler(event, "")
    print(res)
