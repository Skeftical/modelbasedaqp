SELECT departments.department_id, department, count(*) as order_count
FROM order_products, orders, products, departments
WHERE orders.order_id = order_products.order_id
  AND order_products.product_id = products.product_id
  AND products.department_id = departments.department_id
GROUP BY departments.department_id, department
ORDER BY order_count DESC
LIMIT 5;