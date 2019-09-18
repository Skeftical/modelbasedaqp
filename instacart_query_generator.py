import os
import sys
import numpy as np
import pickle
# os.chdir("../")
# print(os.listdir('.'))
if not os.path.exists('input/instacart_queries'):
        # logger.info('creating directory Accuracy')
        print('creating')
        os.makedirs('input/instacart_queries')

query_templates = ['q2.sql', 'q4.sql', 'q6.sql', 'q8.sql','q10.sql','q12.sql']
directory = os.fsencode('instacart_query_templates')
queries = []
for f in os.listdir(directory):
    # print(f)
    with open(os.path.join(directory,f),"r") as sql_query_file:
        sql_query = sql_query_file.read()
        # print(sql_query)
        if os.fsdecode(f) in query_templates:
            # print(sql_query.replace(':d','10'))
            for i in range(50):
                queries.append(sql_query.replace(':d','{}'.format(np.random.normal(8.3510755171755596, 7.1266711612044177))))
        else:
            queries.append(sql_query)
print(len(queries))
with open('input/instacart_queries/queries.pkl', 'wb') as f:
  pickle.dump(queries, f)
