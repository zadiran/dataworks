from v1.measurement.base import base_measurement
from statistics import median

class median_absolute_scaled_error(base_measurement):
    
    def calculate(self, actual_set, forecasted_set):
        denominator = 0
        for x in range(1, len(actual_set)):
            denominator += abs(actual_set[x] - actual_set[x-1])
        denominator /= len(actual_set) - 1

        results = []
        for actual, forecasted in zip(actual_set, forecasted_set):
            results.append(abs(actual - forecasted) / denominator)

        return median(results) ** 0.5


    def get_name(self):
        return 'MdASE'
