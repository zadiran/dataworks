class base_measurement(object):

    def calculate(self, actual_set, forecasted_set):
        raise NotImplementedError('Not implemented. Use derived classes instead of this class')

    def get_name(self):
        return 'Base measurement'

    @staticmethod
    def get_difference(actual_set, forecasted_set):
        diff = []
        for actual, forecasted in zip(actual_set, forecasted_set):
            diff.append(abs(actual - forecasted))

        return diff
        


