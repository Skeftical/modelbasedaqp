-- select sum(add_to_car_order)
-- from order_products
-- where add_to_car_order <= 2
-- group by reordered;
select sum(add_to_car_order)
from order_products
where add_to_car_order <= :d
group by reordered;
