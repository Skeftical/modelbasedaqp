import psycopg2
import argparse
import logging
import os
import pandas as pd
import logging
import time
import sys
import pickle
from psycopg2.extras import NamedTupleCursor
from parser import Parser
if __name__=='__main__':
    print("main executing")
    directory = os.fsencode('/home/fotis/Desktop/tpch_2_17_0/dbgen/tpch_queries_10/')
    conn = psycopg2.connect(host='127.0.0.1',port=5433,dbname='tpch1g',user='analyst',password='analyst',cursor_factory=NamedTupleCursor)
    cur = conn.cursor()
    query_answers_dic = {}
    query_answers_dic['query_name'] = []
    query_answers_dic['time'] = []
    query_names = {}
    i = 0
    for f in os.listdir(directory):
        query_name = os.fsdecode(f).split('.')[0].split('-')[0]
        if query_name not in ['1', '3', '4', '5', '6']:
            continue;
        print("Query Name : {0}".format(query_name))
        with open(os.path.join(directory,f),"r") as sql_query_file:
            sql_query = sql_query_file.read()
            print(sql_query)
            parser = Parser()
            parser.parse(sql_query)
            print("Query Vector {}".format(parser.get_vector()))
            print("Projections {}".format(parser.get_projections()))
            print("Groupby attrs {}".format(parser.get_groupby_attrs()))
