from typing import List
import numpy as np

from sklearn.linear_model import LinearRegression

from models.point_forecast_model import point_forecast_model
from utilities.point import point

class linear_regression_point_forecast_model(point_forecast_model):

    def __init__(self):
        self.lr : LinearRegression

    def fit(self, points : List[point]):
        train_input = []
        train_output = []
        for pt in points:
            train_input.append(pt.input)
            train_output.append(pt.training_output)
            
        self.fit_internal(np.array(train_input), np.array(train_output))

    def fit_internal(self, train_input, train_output):
        
        lr = LinearRegression().fit(train_input, train_output)
        self.lr = lr
        
    def predict(self, inputs):
        return self.lr.predict(np.array(inputs))
