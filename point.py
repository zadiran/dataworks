from typing import Any


class point:
    def __init__(self):
        self.unit : int # dataset, to which point belongs
        self.input : Any # array of data, one-dimensional or two-dimensional
        self.output : Any # RUL value

        self.forecasted_RUL : Any
        self.is_degradation : Any

    def __init__(self, unit : int, input : Any, output : Any):
        self.unit = unit
        self.input = input
        self.output = output
        
        self.forecasted_RUL = 0
        self.is_degradation = False

    def __str__(self):
        _str = f'Point(unit: {self.unit}, input: {self.input}, output: {self.output})'
        return _str
        
    def __repr__(self):
        _str = '_{unit}'#, input: {input}, output: {output})'
        return _str.format(unit = self.unit, input = self.input, output = self.output)