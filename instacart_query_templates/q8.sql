-- select count(*)
-- from order_products
-- where add_to_car_order <= 2
-- group by reordered;
select count(*)
from order_products
where add_to_cart_order <= :d
group by reordered;
