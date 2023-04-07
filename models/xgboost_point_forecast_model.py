from typing import List
import numpy as np

import xgboost as xg

from models.point_forecast_model import point_forecast_model
from utilities.point import point

class xgboost_point_forecast_model(point_forecast_model):

    def __init__(self):
        self.xgb : xg.XGBRegressor

    def fit(self, points : List[point]):
        train_input = []
        train_output = []
        for pt in points:
            train_input.append(pt.input)
            train_output.append(pt.training_output)
            
        self.fit_internal(np.array(train_input), np.array(train_output))

    def fit_internal(self, train_input, train_output):

        xgb_r = xg.XGBRegressor(objective ='reg:linear',
                  n_estimators = 10, seed = 123)
  
        ## Fitting the model
        xgb_r.fit(train_input, train_output)
        self.xgb = xgb_r


    def predict(self, inputs):
        return self.xgb.predict(np.array(inputs))
