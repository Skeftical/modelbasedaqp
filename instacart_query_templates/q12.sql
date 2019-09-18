-- select sum(add_to_car_order)
-- from order_products
-- where add_to_car_order <= 2
-- group by reordered;
select reordered, sum(add_to_cart_order)
from order_products
where add_to_cart_order <= :d
group by reordered;
