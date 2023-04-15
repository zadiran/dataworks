from typing import List
from utils.point import point
from data_processing.point_data_source_rusgidro import point_data_source_rusgidro as pds
import matplotlib.pyplot as plt


#from models.cnn_point_forecast_model import cnn_point_forecast_model as cnn_model
#from models.linear_regression_point_forecast_model import linear_regression_point_forecast_model as lr_model
from models.xgboost_point_forecast_model import xgboost_point_forecast_model as xgb_model

from measurement.absolute import root_mean_square_error as rmse
from measurement.absolute import mean_absolute_error as mae
from measurement.utils import calculate_measurements_for_points

window_size = 7

data = pds().get_data('.local/data/top10pmaxavg_stage_1.csv', window_size)

output_diffs = []
normal_cnt = []
degr_cnt = []

outpt : List[point] = []
    
def top_split_func(combined_train_set, verification_set):
    
    model_for_forecasting = xgb_model()
    model_for_forecasting.fit(combined_train_set)
    model_for_forecasting.predict_points(verification_set)

    outpt.extend(verification_set)

       
def split(data1: List[point]) -> List[List[point]]: 
    output = []
    ln = len(data1)
    numparts = 22

    partition_size = int(ln/numparts)
    
    output = []
    for i in range(0, numparts):
        partition = data1[i * partition_size: (i+1) * partition_size]
        output.append(partition)
    return output            

dataparts = split(data)

pairs = []
for i in range(0, 22):
    pair_1 = dataparts[i]
    pair_2 = []
    for j in range(0, 22):
        if j != i:
            pair_2 = pair_2 + dataparts[j]
    pairs.append([pair_1, pair_2])

for p in pairs:
    top_split_func(p[1], p[0])


print('++++++++++++++++++++++++++++++++')

plt.plot(list(map(lambda x: x.training_output, outpt)))
plt.plot(list(map(lambda x: x.forecasted_output, outpt)))
plt.show()

measurements = calculate_measurements_for_points([rmse(), mae()], outpt)
for m in measurements:
    print(f'{m.name}: {m.value}')


