\copy "region"     from '/home/fotis/Desktop/tpch_2_17_0/verdict_data/tpch1g/region/region.tbl'        DELIMITER '|' CSV;
\copy "nation"     from '/home/fotis/Desktop/tpch_2_17_0/verdict_data/tpch1g/nation/nation.tbl'        DELIMITER '|' CSV;
\copy "customer"   from '/home/fotis/Desktop/tpch_2_17_0/verdict_data/tpch1g/customer/customer.tbl'    DELIMITER '|' CSV;
\copy "supplier"   from '/home/fotis/Desktop/tpch_2_17_0/verdict_data/tpch1g/supplier/supplier.tbl'    DELIMITER '|' CSV;
\copy "part"       from '/home/fotis/Desktop/tpch_2_17_0/verdict_data/tpch1g/part/part.tbl'            DELIMITER '|' CSV;
\copy "partsupp"   from '/home/fotis/Desktop/tpch_2_17_0/verdict_data/tpch1g/partsupp/partsupp.tbl'    DELIMITER '|' CSV;
\copy "orders"     from '/home/fotis/Desktop/tpch_2_17_0/verdict_data/tpch1g/orders/orders.tbl'        DELIMITER '|' CSV;
\copy "lineitem"   from '/home/fotis/Desktop/tpch_2_17_0/verdict_data/tpch1g/lineitem/lineitem.tbl'    DELIMITER '|' CSV;
