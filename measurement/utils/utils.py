from typing import List
from measurement.base import base_measurement
from .calculation_result import calculation_result as cr
from utils import point

def calculate_measurements(measurements : List[base_measurement], expected_values : List[any], forecasted_values: List[any]) -> List[cr]:
    output = []
    for m in measurements:
        result = m.calculate(expected_values, forecasted_values)
        output.append(cr(m.get_name(), result))
    return output

def calculate_measurements_for_points(measurements : List[base_measurement], points : List[point]):
    training_outputs = list(map(lambda x: x.training_output, points))
    forecasted_outputs = list(map(lambda x: x.forecasted_output, points))
    
    return calculate_measurements(measurements, training_outputs, forecasted_outputs)