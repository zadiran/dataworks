from sklearn.linear_model import LinearRegression

from models.forecast_model import forecast_model
from nasa_data_manipulator import nasa_data_manipulator

class baseline_continuous_forecast_model(forecast_model):

    def __init__(self, dm: nasa_data_manipulator):
        self.dm = dm

    def fit(self):
        train_input = self.dm.get_clean_training_input().to_numpy()
        train_output = self.dm.get_proximity_for_arr(self.dm.get_cv_training_output())
        self.fit_internal(train_input, train_output)

    def fit_internal(self, train_input, train_output):
        
        reg = LinearRegression().fit(train_input, train_output)
        self.reg = reg

    def predict(self, inputs):
        return self.reg.predict([inputs])