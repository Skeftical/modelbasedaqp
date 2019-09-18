select sum(add_to_cart_order)
from order_products
group by reordered;
