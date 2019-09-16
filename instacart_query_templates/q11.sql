select sum(add_to_car_order)
from order_products
group by reordered;
