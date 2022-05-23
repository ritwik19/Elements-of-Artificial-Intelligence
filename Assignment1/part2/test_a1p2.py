# !/usr/bin/env python3
# YOU SHOULD NOT MODIFY THIS FILE
# 
# test_a1p2.py version 2021.09.18
#
#Stephen Karukas, Zoher Kachwala, Vrinda Mathur

import route
import pytest

def validate_route(answer, args):
    arg_start, arg_end, arg_cost = args
    assert isinstance(answer, dict), "get_route() is not returning a dictionary"
    assert len(answer) == 5, "Too few parts: returned dictionary should have 5 keys"
    segments, miles = answer['total-segments'], answer['total-miles']
    hours, delivery_hours, route_taken = answer['total-hours'], answer['total-delivery-hours'], answer['route-taken']
    assert isinstance(segments, int), f'{segments} is not an int: total-segments must be int'
    assert segments >= 0, f'{segments} < 0: total-segments must be positive'
    assert isinstance(miles, float), f'{str(miles)} is not a float: total-miles must be float'
    assert miles >= 0, f'{str(miles)} < 0: total-miles must be positive'
    assert isinstance(hours, float), f'{str(hours)} is not an int: total-hours must be int'
    assert hours >= 0, f'{str(hours)} < 0: total-hours must be positive'
    assert isinstance(delivery_hours, float), f'{str(delivery_hours)} is not an float: Any probability must be float'
    assert delivery_hours >= 0, f'{str(delivery_hours)} < 0: total-delivery-hours must be positive'
    assert len(route_taken) == segments, 'Route taken does not correspond to total number of segments'
    assert route_taken[segments - 1][0] == arg_end, f'{route_taken[segments - 1][0]} is not {arg_end}. Not the end-city'
    return segments, miles, hours, delivery_hours

time_ = 300
@pytest.mark.timeout(time_)
def test_part2_case1():
    for script_args in [('Bloomington,_Indiana', 'Indianapolis,_Indiana', x) for x in
                        ('distance', 'segments', 'time', 'delivery')]:
        output = route.get_route(*script_args)
        optimal_ans, calculated = {"segments": 3, "distance": 51.0, "time": 1.07949, "delivery": 1.1364}, {}
        calculated['segments'], calculated['distance'], calculated['time'], calculated['delivery'] = validate_route(output, script_args)
        upper = optimal_ans[script_args[2]] + optimal_ans[script_args[2]]*0.1
        assert calculated[script_args[2]] <= upper, 'Output format is correct but answer is out of range'