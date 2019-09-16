-- select sum(add_to_car_order)
-- from order_products
-- where add_to_car_order <= 2;
select sum(add_to_car_order)
from order_products
where add_to_car_order <= :d;
