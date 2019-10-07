\copy "aisles"     from '/home/fotis/dev_projects/model-based-aqp/input/instacart_2017_05_01/aisles.csv'        DELIMITER ',' CSV HEADER;
\copy "departments"     from '/home/fotis/dev_projects/model-based-aqp/input/instacart_2017_05_01/departments.csv'        DELIMITER ',' CSV HEADER;
\copy "order_products"     from '/home/fotis/dev_projects/model-based-aqp/input/instacart_2017_05_01/order_products__prior.csv'        DELIMITER ',' CSV HEADER;
\copy "orders"     from '/home/fotis/dev_projects/model-based-aqp/input/instacart_2017_05_01/orders.csv'        DELIMITER ',' CSV HEADER;
\copy "products"     from '/home/fotis/dev_projects/model-based-aqp/input/instacart_2017_05_01/products.csv'        DELIMITER ',' CSV HEADER;


copy aisles from 's3://instacart-sample/aisles.csv'
credentials 'aws_iam_role=arn:aws:iam::605088149509:role/myRedshiftRole'
DELIMITER ',' CSV IGNOREHEADER 1 region 'eu-west-1';

copy departments from 's3://instacart-sample/departments.csv'
credentials 'aws_iam_role=arn:aws:iam::605088149509:role/myRedshiftRole'
DELIMITER ',' CSV IGNOREHEADER 1 region 'eu-west-1';

copy order_products from 's3://instacart-sample/order_products__prior.csv' 
credentials 'aws_iam_role=arn:aws:iam::605088149509:role/myRedshiftRole'
DELIMITER ',' CSV IGNOREHEADER 1 region 'eu-west-1';


copy orders from 's3://instacart-sample/orders.csv'
credentials 'aws_iam_role=arn:aws:iam::605088149509:role/myRedshiftRole'
DELIMITER ',' CSV IGNOREHEADER 1 region 'eu-west-1';

copy products from 's3://instacart-sample/products.csv'
credentials 'aws_iam_role=arn:aws:iam::605088149509:role/myRedshiftRole'
DELIMITER ',' CSV IGNOREHEADER 1 region 'eu-west-1';
