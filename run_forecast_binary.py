from typing import List
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score

import matplotlib.pyplot as plt

from data_processing.nasa_data_source import nasa_data_source
from data_processing.nasa_data_manipulator import nasa_data_manipulator
from models.baseline_binary_forecast_model import baseline_binary_forecast_model
from models.baseline_binary_point_forecast_model import baseline_binary_point_forecast_model
from utils.point import point
from data_processing.binary_point_data_source import binary_point_data_source as bpds

from utils.splitter import splitter

window_size = 10

data = bpds().get_data('data/train_FD001.txt', window_size)

def split_func(training_set, verification_set):
    bbpfm = baseline_binary_point_forecast_model()
    bbpfm.fit(training_set)
    bbpfm.predict_points(verification_set)

splitter().run(split_func, data, 5, [])

forecasted = list(map(lambda x: x.forecasted_output, data))
expected = list(map(lambda x: x.training_output, data))

print('ROC/AUC: ' + str(roc_auc_score(forecasted, expected)))
print('F1: ' + str(f1_score(forecasted, expected)))
print('Accuracy: ' + str(accuracy_score(forecasted, expected)))
print('Precision: ' + str(precision_score(forecasted, expected)))

plt.plot(expected[:500], color = 'orange')
plt.scatter(range(0,500), forecasted[:500])
plt.show()
