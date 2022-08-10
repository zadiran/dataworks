from framework.measurement.base import base_measurement

class root_mean_square_percentage_error(base_measurement):
    
    def calculate(self, actual_set, forecasted_set):
        measurement = 0
        for actual, forecasted in zip(actual_set, forecasted_set):
            p = abs(actual - forecasted) / actual
            measurement += (100 * p) ** 2

        return (measurement / len(actual_set)) ** 0.5


    def get_name(self):
        return 'RMSPE'