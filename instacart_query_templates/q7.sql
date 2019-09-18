select reordered, count(*)
from order_products
group by reordered;
