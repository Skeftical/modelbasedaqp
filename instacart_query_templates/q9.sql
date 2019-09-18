select reordered, avg(add_to_cart_order)
from order_products
group by reordered;
