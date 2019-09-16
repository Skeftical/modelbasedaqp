-- select count(*)
-- from order_products
-- where add_to_car_order <= 2;

select count(*)
from order_products
where add_to_car_order <= :d;
