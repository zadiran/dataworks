from typing import List
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPool1D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import numpy as np

from forecast_model import forecast_model
from point import point

class cnn_point_forecast_model(forecast_model):

    def __init__(self):
        self.cnn : Sequential

    def fit(self, points : List[point]):
        train_input = []
        train_output = []
        for pt in points:
            train_input.append(pt.input)
            train_output.append(pt.output)
            
        #print(train_input[0:5])
        #print(train_output[0:5])
        self.fit_internal(np.array(train_input), np.array(train_output))

    def fit_internal(self, train_input, train_output):
        
        path = 'model/regression_model.cnn.h5'

        cnn = Sequential([
            Conv1D(filters= 64, kernel_size= 3, activation='relu'),
            MaxPool1D(pool_size = 2),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(1)
        ])
        cnn.compile(optimizer="rmsprop", loss = "mean_squared_error", metrics = "mae")
        cnn.fit(train_input, train_output, epochs=35, batch_size = 200, verbose = 0, validation_split= 0.2, use_multiprocessing=True, callbacks= [
            EarlyStopping(patience=10, mode = 'min'),
            ModelCheckpoint(path, save_best_only=True, mode = 'min')
        ])

        self.cnn = cnn

    def predict(self, inputs):
        return self.cnn.predict(np.array(inputs))
    
    def predict_points(self, points):
        inputs = []
        for pt in points:
            inputs.append(pt.input)

        forecasts = self.predict(inputs)
        for p, f in zip(points, forecasts):
            p.forecasted_RUL = f

        return points