CREATE TABLE aisles (
  aisleid int PRIMARY KEY,
  aisle varchar(64),
);

CREATE TABLE departments (
  departmentid int PRIMARY KEY,
  department varchar(64)
);

CREATE TABLE order_products (
  order_id int,
  product_id int,
  add_to_cart_order int,
  reordered int,
  PRIMARY KEY(order_id, product_id)
);

CREATE TABLE orders (
  order_id int,
  user_id int,
  eval_set varchar(16),
  order_number int,
  order_dow int,
  order_hour_of_day int,
  days_since_prior_order float,
  PRIMARY KEY(order_id, user_id)
);
