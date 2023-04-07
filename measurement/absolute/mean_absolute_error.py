from measurement.base import base_measurement

class mean_absolute_error(base_measurement):
    
    def calculate(self, actual_set, forecasted_set):
        measure = 0
        for actual, forecasted in zip(actual_set, forecasted_set):
            measure += abs(actual - forecasted)
        return measure / len(actual_set)

    def get_name(self):
    	return 'MAE'