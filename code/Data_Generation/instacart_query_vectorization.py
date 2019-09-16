import os
import sys
os.chdir('../../')
import pickle
import psycopg2

queries = []
with open('input/instacart_queries/queries.pkl','rb') as f:
    queries = pickle.load(f)
