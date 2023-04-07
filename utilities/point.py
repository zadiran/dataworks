from typing import Any


#=======================================================================================
# point
#=======================================================================================
# Класс для хранения данных точки временного ряда. 
#---------------------------------------------------------------------------------------
# Поля:
#   unit - номер экземпляра оборудования, юнит
#   input - входные данные точки: число, одномерный или двумерный массив
#   training_output - выходное значение RUL для обучения 
#   forecasted_output - спрогнозированное выходное значение RUL
#   is_degradation - флаг, принадлежит ли данная точка деградационному процессу
#=======================================================================================
class point:
    def __init__(self):
        self.unit : int 
        self.input : Any 
        self.training_output : Any 

        self.forecasted_output : Any
        self.is_degradation : Any

    def __init__(self, unit : int, input : Any, training_output : Any):
        self.unit = unit
        
        self.input = input

        self.training_output = training_output
        self.forecasted_output = 0
        
        self.is_degradation = False

    def __str__(self):
        _str = f'Point(unit: {self.unit}, input: {self.input}, training_output: {self.training_output})'
        return _str
        
    def __repr__(self):
        _str = 'unit: {unit}, input: {input}, training_output: {training_output})'
        return _str.format(unit = self.unit, input = self.input, training_output = self.training_output)