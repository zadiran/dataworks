from measurement.base import base_measurement
from statistics import median


class root_median_square_percentage_error(base_measurement):
    
    def calculate(self, actual_set, forecasted_set):
        results = []

        for actual, forecasted in zip(actual_set, forecasted_set):
            p = abs(actual - forecasted) / actual
            results.append((100 * p) ** 2)

        return median(results) ** 0.5
    
    def get_name(self):
        return 'RMdSPE'

