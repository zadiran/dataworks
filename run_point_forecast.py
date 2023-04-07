from typing import List
from utilities.point import point
from point_data_source import point_data_source as pds
from utilities.splitter import splitter 

from data_source import data_source

from scipy.stats import kruskal
from models.cnn_point_forecast_model import cnn_point_forecast_model as cpfm

window_size = 50

data = pds().get_data('train_FD001.txt', window_size)

class virtual_data_source(data_source):
    def get_data():
        return data

output_diffs = []
normal_cnt = []
degr_cnt = []
    
def top_split_func(combined_train_set, verification_set):
    splt2 = splitter()

    print('###########################################')
    print('# top split')
    print('###########################################')
    # models - array of [model_for_splitting, model_for_forecasting]
    models = splt2.run(prepare_models, combined_train_set, 3, verification_set)
    
    print ('=== models learned ===')
    for model_set in models:
        degr_data = get_degr_data(verification_set, model_set[0])
        results = model_set[1].predict_points(degr_data)
        for r in results:
            output_diffs.append(abs(r.training_output - r.forecasted_output))


# returns [model_for_splitting, model_for_forecasting]
def prepare_models(long_set, short_set):
    model_for_splitting = cpfm()
    model_for_splitting.fit(short_set)
    
    degr_data = get_degr_data(long_set, model_for_splitting)
    model_for_forecasting = cpfm()
    model_for_forecasting.fit(degr_data)

    return [model_for_splitting, model_for_forecasting]



def get_degr_data(dataset : List[point], cnn : cpfm):
    for pt in dataset:
        pt.is_degradation = False
        pt.forecasted_output = 0

    inputs = []
    for pt in dataset:
        inputs.append(pt.input)

    forecasts = cnn.predict(inputs)
    for p, f in zip(dataset, forecasts):
        p.forecasted_output = f
    
    unique_units = list(set(map(lambda x: x.unit, dataset)))
    print(f'unique units: {unique_units}')

    for u in unique_units:
        unit_points = [x for x in dataset if x.unit == u]
        
        degradation_start_index = hmhm(unit_points)
        for i, up in enumerate(unit_points):
            if i >= degradation_start_index:
                up.is_degradation = True        

    output = []
    for pt in dataset:
        if pt.is_degradation:
            output.append(pt)
    
    return output

def hmhm(points : List[point]):
    rul_values = []
    for pt in points:
        rul_values.append(pt.forecasted_RUL)

    kruskal_values = []
    diff = 5

    for i, rul in enumerate(rul_values):
        if i + window_size - 1 >= len(rul_values) or i < 2 * diff:
            kruskal_values.append(0)
        else:
            w1 = rul_values[i-diff: i-diff + window_size]
            w2 = rul_values[i: i + window_size]
            kruskal_values.append(kruskal(w1, w2).pvalue * 100)
    
    print('===================================================')
    print('kruskal_values')
    print('===================================================')
    print(kruskal_values)
    print('===================================================')
    
    degr_start_index = 0
    consequent_ge_5 = 0

    diff2 = 100
    for i, kw_val in enumerate(kruskal_values):
        if kw_val < 5:
            consequent_ge_5 = 0
        else:
            consequent_ge_5 = consequent_ge_5 + 1

        if consequent_ge_5 >= diff2:
            degr_start_index = i
            break
    
    normal_cnt.append(degr_start_index)
    degr_cnt.append(len(kruskal_values) - degr_start_index)
    return degr_start_index
        
            
            
splt_1 = splitter()
splt_1.run(top_split_func, data, 4, [])

print(f'MAE HMHM: {sum(output_diffs) / len(output_diffs)}')

nc_sum = sum(normal_cnt)
d_sum = sum(degr_cnt)

print(f'Total normal operation: {nc_sum} out of {nc_sum + d_sum} or {100 * nc_sum / (nc_sum + d_sum)}%')
print(f'Total degradation: {d_sum} out of {nc_sum + d_sum} or {100 * d_sum / (nc_sum + d_sum)}%')

