import json
# import boto3
import pickle
import numpy as np
from flask import Flask, request, json

with open('distinct_attribute_catalogue.pkl', 'rb') as f:
    distinct_attr = pickle.load(f)

with open('model_catalogue.pkl', 'rb') as f:
    model_catalogue = pickle.load(f)

with open('labels_catalogue.pkl', 'rb') as f:
    labels_catalogue = pickle.load(f)




@app.route('/', methods=['POST'])
def index():
    # Parse request body for model input
    event = request.get_json(silent=True)
    proj_dict = event['projections']
    groups = event['groups']
    filters = event['filters']

    # Load model
    res = {}
    for p in proj_dict:
        res[p] = []
        est = model_catalogue[p]
        if len(groups)>0:
            for g in groups:
                gvalues = distinct_attr[g]
                print("length of groupby values {}".format(len(gvalues)))
                res[g] = gvalues
                filters[g+'_lb'] = [labels_catalogue.get(gval,np.nan) for gval in gvalues]
                res[p]+=est.predict_many(filters).tolist()
        else:
            res[p].append(est.predict_one(filters))

    result = {'result': res}
return json.dumps(result)

if __name__ == '__main__':
    # listen on all IPs
    app.run(host='0.0.0.0')

    # res = lambda_handler(event, "")
    # print(res)
