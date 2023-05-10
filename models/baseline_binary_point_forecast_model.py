from typing import List
import numpy as np
from sklearn.linear_model import LogisticRegression

from models.point_forecast_model import point_forecast_model
from data_processing.nasa_data_manipulator import nasa_data_manipulator
from utils.point import point


class baseline_binary_point_forecast_model(point_forecast_model):

    def __init__(self):
        self.lr : LogisticRegression
    
    def fit(self, points : List[point]):
        train_input = []
        train_output = []
        for pt in points:
            train_input.append(pt.input)
            train_output.append(pt.training_output)
            
        self.fit_internal(np.array(train_input), np.array(train_output))

    def fit_internal(self, train_input, train_output):
        lr = LogisticRegression(solver='liblinear').fit(train_input, train_output)
        self.lr = lr

    def predict(self, inputs):
        return self.lr.predict(inputs)
    

        
    