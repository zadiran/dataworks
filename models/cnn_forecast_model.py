from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPool1D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from models.forecast_model import forecast_model
from data_processing.nasa_2d_data_manipulator import nasa_2d_data_manipulator

class cnn_forecast_model(forecast_model):

    def __init__(self, dm: nasa_2d_data_manipulator):
        self.dm : nasa_2d_data_manipulator = dm
        self.cnn : Sequential

    def fit(self):
        train_input = self.dm.get_training_input()
        train_output = self.dm.get_training_output()
        self.fit_internal(train_input, train_output)

    def fit_internal(self, train_input, train_output):
        
        path = '.local/model/regression_model.cnn.h5'

        cnn = Sequential([
            Conv1D(filters= 64, kernel_size= 3, activation='relu'),
            MaxPool1D(pool_size = 2),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(1)
        ])
        cnn.compile(optimizer="rmsprop", loss = "mean_squared_error", metrics = "mae")
        cnn.fit(train_input, train_output, epochs=35, batch_size = 200, verbose = 2, validation_split= 0.2, use_multiprocessing=True, callbacks= [
            EarlyStopping(patience=10, mode = 'min'),
            ModelCheckpoint(path, save_best_only=True, mode = 'min')
        ])

        self.cnn = cnn

    def predict(self, inputs):
        return self.cnn.predict(inputs)