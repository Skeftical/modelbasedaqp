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
np.random.seed(15)

query_templates = ['q2.sql', 'q4.sql', 'q6.sql', 'q8.sql','q10.sql','q12.sql']
directory = os.fsencode('instacart_query_templates')
NUMBER_OF_QUERIES = 1000
queries = []
for f in os.listdir(directory):
    # print(f)
    with open(os.path.join(directory,f),"r") as sql_query_file:
        sql_query = sql_query_file.read()
        # print(sql_query)
        if os.fsdecode(f) in query_templates:
            # print(sql_query.replace(':d','10'))
            for i in range(NUMBER_OF_QUERIES):
                queries.append((os.fsdecode(f),sql_query.replace(':d','{}'.format(np.random.normal(8.3510755171755596, 7.1266711612044177)))))
        elif os.fsdecode(f)=='q14.sql':
            for i in range(NUMBER_OF_QUERIES//5):
                queries.append((os.fsdecode(f),sql_query.replace(':d','{}'.format(np.random.randint(0,7)))))
        elif os.fsdecode(f) not in ['q1.sql','q3.sql', 'q5.sql']:
            for i in range(NUMBER_OF_QUERIES):
                queries.append((os.fsdecode(f),sql_query))
print(len(queries))
with open('input/instacart_queries/queries-{}.pkl'.format(NUMBER_OF_QUERIES), 'wb') as f:
  pickle.dump(queries, f)
queries = []
for f in os.listdir(directory):
    # print(f)
    with open(os.path.join(directory,f),"r") as sql_query_file:
        sql_query = sql_query_file.read()
        # print(sql_query)
        if os.fsdecode(f) in query_templates:
            # print(sql_query.replace(':d','10'))
            for i in range(np.ceil(NUMBER_OF_QUERIES/4).astype(int)):
                queries.append((os.fsdecode(f),sql_query.replace(':d','{}'.format(np.random.normal(8.3510755171755596, 7.1266711612044177)))))
        elif os.fsdecode(f)=='q14.sql':
            for i in range(np.ceil(NUMBER_OF_QUERIES/20).astype(int)):
                queries.append((os.fsdecode(f),sql_query.replace(':d','{}'.format(np.random.randint(0,7)))))
        else:
            for i in range(np.ceil(NUMBER_OF_QUERIES/4).astype(int)):
                queries.append((os.fsdecode(f),sql_query))
print(len(queries))
with open('input/instacart_queries/queries-test-{}.pkl'.format(NUMBER_OF_QUERIES), 'wb') as f:
  pickle.dump(queries, f)
