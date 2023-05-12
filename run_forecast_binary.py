from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score

import matplotlib.pyplot as plt

from models.baseline_binary_point_forecast_model import baseline_binary_point_forecast_model
from data_processing import csv_data_source
from data_processing.configurable import configurable_data_manipulator
from data_processing.configurable.stages.pre_point_conversion import drop_columns, normalize
from data_processing.configurable.stages.point_conversion import convert_to_1d_input
from data_processing.configurable.stages.post_point_conversion import convert_to_binary_output


from utils.splitter import splitter

window_size = 10

raw_data = csv_data_source().get_data('data/train_FD001.csv', ';')

cdm = configurable_data_manipulator('.local/cache/all_input_binary.pickle')

cdm.add_pre_point_conversion_stage(drop_columns(['s3', 's4', 's8', 's9', 's13', 's19', 's21', 's22']))
cdm.add_pre_point_conversion_stage(normalize())
cdm.set_point_conversion_stage(convert_to_1d_input(window_size))
cdm.add_post_point_conversion_stage(convert_to_binary_output(10))

data = cdm.get_processed_data(raw_data)

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
