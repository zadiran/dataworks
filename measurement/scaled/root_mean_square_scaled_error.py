import measurement.base import base_measurement

class root_mean_square_scaled_error(base_measurement):
    
    def calculate(self, actual_set, forecasted_set):
        denominator = 0
        for x in range(1, len(actual_set)):
            denominator += abs(actual_set[x] - actual_set[x-1])
        denominator /= len(actual_set) - 1

        measurement = 0

        for actual, forecasted in zip(actual_set, forecasted_set):
            measurement += (abs(actual - forecasted) / denominator) ** 2

        return (measurement / len(actual_set)) ** 0.5
        
    def get_name(self):
        return 'RMSSE'