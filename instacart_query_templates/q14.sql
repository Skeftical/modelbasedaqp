-- SELECT product_name, count(*) as order_count
-- FROM order_products, orders, products
-- WHERE orders.order_id = order_products.order_id
--   AND order_products.product_id = products.product_id
--   AND (order_dow = 0 OR order_dow = 1)
-- GROUP BY product_name
-- ORDER BY order_count DESC
-- LIMIT 5;
SELECT product_name, count(*) as order_count
FROM order_products, orders, products
WHERE orders.order_id = order_products.order_id
  AND order_products.product_id = products.product_id
  AND (order_dow = :d OR order_dow = :d1)
GROUP BY product_name
ORDER BY order_count DESC
LIMIT 5;
