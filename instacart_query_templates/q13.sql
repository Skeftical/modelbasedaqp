select order_hour_of_day, count(*)
from orders
group by order_hour_of_day
order by order_hour_of_day;
