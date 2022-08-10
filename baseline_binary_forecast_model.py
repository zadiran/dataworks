from sklearn.linear_model import LogisticRegression

from forecast_model import forecast_model
from nasa_data_manipulator import nasa_data_manipulator


class baseline_binary_forecast_model(forecast_model):

    def __init__(self, dm: nasa_data_manipulator):
        self.dm = dm

    def fit(self):
        training_input = self.dm.get_clean_training_input().to_numpy()
        training_output = self.dm.get_logical_proximity(self.dm.get_cv_training_output(), 10)

        self.fit_internal(training_input, training_output)

    def fit_internal(self, train_input, train_output):
        reg = LogisticRegression(solver='liblinear').fit(train_input, train_output)
        self.reg = reg

    def predict(self, inputs):
        return self.reg.predict([inputs])
        