import json
import requests

event = {}
event['projections'] = ['count']
event['groups'] = ['product_name']
event['filters']= {'order_dow_lb': 4.0, 'order_dow_ub': 4.0}

r = requests.post("http://localhost:5000", json=event)
print(r.status_code, r.reason)
