from data_processing import csv_data_source, point_data_source
from data_processing.configurable.configurable_data_manipulator import configurable_data_manipulator
from data_processing.configurable.stages.point_conversion.convert_to_2d_input import convert_to_2d_input
from data_processing.configurable.stages.pre_point_conversion.drop_columns import drop_columns
from data_processing.configurable.stages.pre_point_conversion.normalize import normalize

window_size = 5

data = csv_data_source().get_data('data/train_FD001.csv', ';')

cdm = configurable_data_manipulator()

cdm.add_pre_point_conversion_stage(drop_columns(['s3', 's4', 's8', 's9', 's13', 's19', 's21', 's22']))
cdm.add_pre_point_conversion_stage(normalize())

cdm.set_point_conversion_stage(convert_to_2d_input(window_size))


processed_data = cdm.get_processed_data(data)
print(processed_data[0].unit)



data2 = point_data_source().get_data('data/train_FD001.txt', window_size)
print(data2[0].unit)