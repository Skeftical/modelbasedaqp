CREATE TABLE IF NOT EXISTS aisles (
  aisleid int,
  aisle varchar(64)
);

CREATE TABLE IF NOT EXISTS departments (
  departmentid int,
  department varchar(64)
);

CREATE TABLE IF NOT EXISTS order_products (
  order_id int,
  product_id int,
  add_to_cart_order int,
  reordered int,
  PRIMARY KEY(order_id, product_id)
);

CREATE TABLE IF NOT EXISTS orders (
  order_id int,
  user_id int,
  eval_set varchar(16),
  order_number int,
  order_dow int,
  order_hour_of_day int,
  days_since_prior_order float,
);

CREATE TABLE IF NOT EXISTS products (
    product_id int,
    product_name varchar(512),
    aisle_id int,
    department_id int
);
