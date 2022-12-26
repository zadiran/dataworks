from v1.measurement.base import base_measurement

class mean_square_error(base_measurement):
    
    def calculate(self, actual_set, forecasted_set):
        measure = 0
        for actual, forecasted in zip(actual_set, forecasted_set):
            measure += (actual - forecasted) ** 2
        return measure / len(actual_set)

    def get_name(self):
        return 'MSE'



