from typing import List
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPool1D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np



from models.point_forecast_model import point_forecast_model
from utils.point import point

class cnn_point_forecast_model(point_forecast_model):

    def __init__(self):
        self.cnn : Sequential

    def fit(self, points : List[point]):
        train_input = []
        train_output = []
        for pt in points:
            train_input.append(pt.input)
            train_output.append(pt.training_output)
            
        self.fit_internal(np.array(train_input), np.array(train_output))

    def fit_internal(self, train_input, train_output):
        
        path = 'local/model/regression_model.cnn.h5'

        cnn = Sequential([
            Conv1D(filters= 64, kernel_size= 3, activation='relu'),
            MaxPool1D(pool_size = 2),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(1)
        ])
        cnn.compile(optimizer="rmsprop", loss = "mean_squared_error", metrics = "mae")
        cnn.fit(train_input, train_output, epochs=35, batch_size = 200, verbose = 1, validation_split= 0.2, use_multiprocessing=True, shuffle= True, callbacks= [
           EarlyStopping(patience=10, mode = 'min'),
           ModelCheckpoint(path, save_best_only=True, mode = 'min')
        ])

        self.cnn = cnn


    def predict(self, inputs):
        return self.cnn.predict(np.array(inputs))
