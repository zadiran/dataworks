from typing import List

from models.forecast_model import forecast_model
from utils.point import point

class point_forecast_model(forecast_model):

    def predict_points(self, points : List[point]):
        inputs = []
        for pt in points:
            inputs.append(pt.input)

        forecasts = self.predict(inputs)
        for p, f in zip(points, forecasts):
            p.forecasted_output = f

        return points