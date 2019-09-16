import sqlparse
from sqlparse.sql import Where, TokenList, Function
from sqlparse.tokens import Name, Keyword, DML, Wildcard, Comparison
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
                if attr not in self.query_vec:
                    self.query_vec[attr]={}
                if lit_dir=='bi':
                    self.query_vec[attr]['lb']=lit
                    self.query_vec[attr]['ub']=lit
                    continue;
                self.query_vec[attr][lit_dir] = lit

    def __projections(self,token, tokenlist):
        idx = tokenlist.token_index(token)
        afs_list_idx, afs = tokenlist.token_next(idx, skip_ws=True, skip_cm=True)
        afs_list = TokenList(list(afs.flatten()))
        for af in afs_list: # Get AFs

            if af.value.lower() in ['avg','count','sum','min','max']:
                if af not in self.afs_dic:
                    self.afs_dic[af.value] = []
                af_idx = afs_list.token_index(af)
                punc_idx, _ = afs_list.token_next(af_idx, skip_ws=True, skip_cm=True)
                attr_idx, attr = afs_list.token_next(punc_idx, skip_ws=True, skip_cm=True)
                if attr.ttype is not Wildcard:
                    self.afs_dic[af.value].append(attr.value)

    def __groupbyattrs(self, token, tokenlist):
        g_index = tokenlist.token_index(token)
        attr_idx , attr = tokenlist.token_next(g_index)
        for g_attr in attr.flatten():
            if g_attr.ttype is Name:
                self.groupby_attrs.append(g_attr.value)

    def get_groupby_attrs(self):
        return self.groupby_attrs

    def get_projections(self):
        return self.afs_dic

    def get_vector(self):
        return self.query_vec

    def __init__(self):
        self.query_vec = {} # {attr : {'lb' : val, 'ub': val}}
        self.afs_dic = {} # {'af' : ['attr1','attr2']}
        self.groupby_attrs = []


if __name__=='__main__':
    parser = Parser()
    parser.parse(sql2)
    print(parser.get_vector())
    print(parser.get_projections())
    print(parser.get_groupby_attrs())
