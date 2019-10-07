import sqlparse
from sqlparse.sql import Where, TokenList, Function
from sqlparse.tokens import Name, Keyword, DML, Wildcard, Comparison
import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
sql = """select avg(add_to_car_order), count(*) tags, sum(cart)
from order_products
where add_to_cart_order <= 2 OR add_to_cart_order>0 AND packet=4.555
group by reordered;"""

sql2 = """
SELECT product_name, count(*) as order_count
FROM order_products, orders, products
WHERE orders.order_id = order_products.order_id
  AND order_products.product_id = products.product_id
  AND (order_dow = 0 OR order_dow = 1)
GROUP BY product_name
ORDER BY order_count DESC
LIMIT 5;
"""
class Parser():

    def parse(self,sql_stmt):
        self.parsed = sqlparse.parse(sql_stmt)
        for t in self.parsed[0]:
            if isinstance(t,Where):
                self.__vectorize(t)
            if t.ttype is DML and t.value.lower()=='select':
                self.__projections(t,self.parsed[0])
            if t.ttype is Keyword and t.value.lower()=='group by':
                self.__groupbyattrs(t, self.parsed[0])


    def __vectorize(self,tokenlist):
        token_list = TokenList(list(tokenlist.flatten()))
        # print(token_list.tokens)
        for x in token_list:
            if x.ttype is Comparison:
                idx_comp_op = token_list.token_index(x) #Index of comparison operator
                attr = token_list.token_prev(idx_comp_op,skip_ws=True, skip_cm=True)[1].value#Name of the attribute
                print(attr)
                comp_op = x
                # print(comp_op)
                if comp_op.value =='<' or comp_op.value=='<=':
                    lit_dir = 'ub'
                elif comp_op.value == '>' or comp_op.value=='>=':
                    lit_dir = 'lb'
                else:
                    lit_dir = 'bi'
                # print(lit_dir)
                try :
                    lit = float(token_list.token_next(idx_comp_op, skip_ws=True, skip_cm=True)[1].value) #literal value
                except ValueError:
                    print("Possible join, skipping")
                    continue;
                # print(lit)
                if lit_dir=='bi':
                    self.query_vec['_'.join([attr,'lb'])] = lit
                    self.query_vec['_'.join([attr, 'ub'])] = lit
                    continue;
                self.query_vec['_'.join([attr,lit_dir])] = lit #lit_dir is either lb or ub

    def __projections(self,token, tokenlist):
        idx = tokenlist.token_index(token)
        afs_list_idx, afs = tokenlist.token_next(idx, skip_ws=True, skip_cm=True)
        afs_list = TokenList(list(afs.flatten()))
        for af in afs_list: # Get AFs

            if af.value.lower() in ['avg','count','sum','min','max']:
                # if af not in self.afs_dic:
                #     self.afs_dic[af.value] = []
                af_idx = afs_list.token_index(af)
                punc_idx, _ = afs_list.token_next(af_idx, skip_ws=True, skip_cm=True)
                attr_idx, attr = afs_list.token_next(punc_idx, skip_ws=True, skip_cm=True)
                if attr.ttype is not Wildcard:
                    self.afs.append('_'.join([af.value, attr.value]))
                else:
                    self.afs.append(af.value)

    def __groupbyattrs(self, token, tokenlist):
        g_index = tokenlist.token_index(token)
        attr_idx , attr = tokenlist.token_next(g_index)
        for g_attr in attr.flatten():
            if g_attr.ttype is Name:
                self.groupby_attrs.append(g_attr.value)

    def get_groupby_attrs(self):
        return self.groupby_attrs

    def get_projections(self):
        return self.afs

    def get_vector(self):
        return self.query_vec

    def __init__(self):
        self.query_vec = {} # {attr : {'lb' : val, 'ub': val}}
        self.afs = [] # {'af' : ['attr1','attr2']}
        self.groupby_attrs = []


class QueryVectorizer():

    def __trickle_down(self, length):

        for k in self.__internal_dict:
            k_length = len(self.__internal_dict[k])
            if k_length<length and k_length!=0:
                self.__internal_dict[k]+=[self.__internal_dict[k][-1]]*(length-k_length)

    def insert(self,key, value):
        """
        Insert using key(=attribute) and value(s)
        ------------------------------------------
        key : str
        value : int,str,float,list
        """
        #Whenever there is a list it must be a groupby attribute
        if isinstance(value, list):
            self.__internal_dict[key]+=value
            self.__trickle_down(len(self.__internal_dict[key]))
        else:
            self.__internal_dict[key].append(value)
        listlength = len(self.__internal_dict[key])
        self.__max_row_size = listlength if listlength > self.__max_row_size else self.__max_row_size

    def to_dense(self):
        if self.__sparse_matrix is not None:
            return self.__sparse_matrix.todense()
        else:
            self.to_matrix()
            return self.to_dense()

    def to_dataframe(self):
        if self.__sparse_matrix is not None:
            self.inverse_attr_str_mapper = dict([(value, key) for key, value in self.attr_str_mapper.items()])
            df = pd.DataFrame(self.to_dense(), columns=self.__column_names)
            for attr in self.attrs_with_str:
                df[attr] = df[attr].replace(self.inverse_attr_str_mapper)
            df = df.replace({0:np.nan, -10: 0})

            return df
        else:
            self.to_matrix()
            return self.to_dataframe()

    def to_matrix(self):
        row_ind = []
        col_ind = []
        data = []
        for i,attr in enumerate(self.__internal_dict):
            for j,val in enumerate(self.__internal_dict[attr]):
                col_ind.append(i)
                row_ind.append(j)
                if val==0:
                    val = -10
                if isinstance(val, str):
                    self.attrs_with_str.add(attr)
                    val = self.attr_str_mapper.setdefault(val, len(self.attr_str_mapper))

                data.append(val)
        self.__sparse_matrix = csr_matrix((data, (row_ind, col_ind)),shape=(self.__max_row_size,self.__column_size))
        return self.__sparse_matrix

    def get_column_names(self):
        return self.__column_names

    def _get_internal_representation(self):
        return self.__internal_dict

    def __init__(self, attributes, SET_OWN=False):
        self.__internal_dict = {}
        self.attrs_with_str = set()
        self.attr_str_mapper = {}
        self.__max_row_size = 0
        self.__sparse_matrix = None
        if not SET_OWN:
            for k in attributes:
                self.__internal_dict['_'.join([k,'lb'])] = []
                self.__internal_dict['_'.join([k,'ub'])] = []
        else:
            for k in attributes:
                self.__internal_dict[k] = []
        self.__column_size = len(self.__internal_dict)
        self.__column_names = self.__internal_dict.keys()


if __name__=='__main__':
    # parser = Parser()
    # parser.parse(sql)
    # print(parser.get_vector())
    # print(parser.get_projections())
    # print(parser.get_groupby_attrs())
    qv = QueryVectorizer(['a1','a2','a3'])
    qv.insert('a1_lb',10)
    qv.insert('a2_lb',['a','b','c'])
    qv.insert('a2_lb',['a','b','c'])
    qv.insert('a3_ub', [0,1])
    print(qv._get_internal_representation())
    print(qv.to_matrix())
    print(qv.to_dense())
    print(qv.to_dataframe())
