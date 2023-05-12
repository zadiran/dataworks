from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score

import matplotlib.pyplot as plt
from measurement.utils import calculate_measurements_for_points

from models.linear_regression_point_forecast_model import linear_regression_point_forecast_model
from data_processing import csv_data_source
from data_processing.configurable import configurable_data_manipulator
from data_processing.configurable.stages.pre_point_conversion import drop_columns, normalize
from data_processing.configurable.stages.point_conversion import convert_to_1d_input

from measurement.absolute import root_mean_square_error as rmse
from measurement.absolute import mean_absolute_error as mae

from utils.splitter import splitter

window_size = 10

raw_data = csv_data_source().get_data('data/train_FD001.csv', ';')

cdm = configurable_data_manipulator('.local/cache/all_input_continuous.pickle')

cdm.add_pre_point_conversion_stage(drop_columns(['s3', 's4', 's8', 's9', 's13', 's19', 's21', 's22']))
cdm.set_point_conversion_stage(convert_to_1d_input(window_size))

data = cdm.get_processed_data(raw_data)

def split_func(training_set, verification_set):
    bbpfm = linear_regression_point_forecast_model()
    bbpfm.fit(training_set)
    bbpfm.predict_points(verification_set)

splitter().run(split_func, data, 5, [])

plt.plot(list(map(lambda x: x.training_output, data)))
plt.plot(list(map(lambda x: x.forecasted_output, data)))
plt.show()

measurements = calculate_measurements_for_points([rmse(), mae()], data)
for m in measurements:
    print(f'{m.name}: {m.value}')


