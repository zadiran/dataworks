import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from data_processing.nasa_data_source import nasa_data_source
from data_processing.nasa_data_manipulator import nasa_data_manipulator
from models.baseline_continuous_forecast_model import baseline_continuous_forecast_model

from data_processing.nasa_2d_data_manipulator import nasa_2d_data_manipulator
from models.cnn_forecast_model import cnn_forecast_model

from measurement.absolute import root_mean_square_error as rmse
from measurement.absolute import mean_absolute_error as mae

from scipy.stats import kruskal

# CNN
nds = nasa_data_source('data/train_FD001.txt')
dm = nasa_2d_data_manipulator(nds)

models = []
models2 = []
for i in range(0,5):
    fm = cnn_forecast_model(dm)
    
    dm.set_cv_range(i * 19 + 1, (i+1) * 19 + 1)
    fm.fit()
    models.append(fm)

    # for j in range(1, 1):
    #     fm1 = cnn_forecast_model(dm)
    #     fm1.fit()
    #     models2.append(fm1)

dm.set_cv_range(96, 101)

input_data = dm.get_testing_input()
print('len input data: ' + str(len(input_data)))

output_data = dm.get_testing_output()
print('len output_data after proximity: ' + str(len(output_data)))
#print(output_data)
print('====================================')

units = dm.get_units()
print('len units: ' + str(len(units)))
unitsmagnified = []
for u in units: 
    unitsmagnified.append((u - 95) * 50)

forecasts = []
for model in models:
    forecast = model.predict(input_data)
    forecasts.append(forecast)

    
# forecasts2 = []
# for model in models2:
#     forecast = model.predict(input_data)
#     forecasts2.append(forecast)

avg_forecast = []
for i in range(0, len(input_data)):
    a = []
    for j in range(0, len(forecasts)):
        a.append(forecasts[j][i])

    avg_forecast.append(np.mean(a))
    
# avg_forecast2 = []
# for i in range(0, len(input_data)):
#     a = []
#     for j in range(0, len(forecasts2)):
#         a.append(forecasts2[j][i])

#     avg_forecast2.append(np.mean(a))

print('CNN total rmse ' + str(rmse().calculate(avg_forecast, output_data)))
print('CNN total mae  ' + str(mae().calculate(avg_forecast, output_data)))


# print('CNN total rmse 100 models ' + str(rmse().calculate(avg_forecast2, output_data)))
# print('CNN total mae 100 models  ' + str(mae().calculate(avg_forecast2, output_data)))

print(kruskal(output_data, avg_forecast).pvalue)
# test

df = pd.DataFrame({'unit': units, 'expected': output_data, 'forecasted': avg_forecast, 'kruskal': None, 'kruskal2' : None})

setsize = 50
original_set = df.iloc[0:setsize]['forecasted'].to_numpy()
need_reset = True

for i, row in df.iterrows():
    if i + setsize - 1 >= df.shape[0]:
         df.at[i, 'kruskal'] = 0
    else:
        start_unit = df.at[i, 'unit']
        end_unit = df.at[i + setsize - 1, 'unit']
        if start_unit == end_unit:
            need_reset = True
            df.at[i, 'kruskal'] = kruskal(original_set, df.iloc[i: i + setsize]['forecasted'].to_numpy()).pvalue * 100
        else:
            if need_reset:
                need_reset = False
                df.at[i, 'kruskal'] = 0
                original_set = df.iloc[i + setsize - 1: i + setsize * 2 - 1]['forecasted'].to_numpy()

#test2 
diff = 10
for i, row in df.iterrows():
    if i + setsize - 1 >= df.shape[0] or i < diff:
         df.at[i, 'kruskal2'] = 0
    else:
        start_unit = df.at[i-diff, 'unit']
        end_unit = df.at[i + setsize - 1, 'unit']
        if start_unit == end_unit:
            df.at[i, 'kruskal2'] = kruskal(df.iloc[i-diff: i-diff + setsize]['forecasted'].to_numpy(), df.iloc[i: i + setsize]['forecasted'].to_numpy()).pvalue * 100
        else:
            df.at[i, 'kruskal2'] = 0
            original_set = df.iloc[i + setsize - 1: i + setsize * 2 - 1]['forecasted'].to_numpy()
#print(df)

df['res'] = None
diff2 = 25
for i, row in df.iterrows():
    if i < diff2:
         df.at[i, 'res'] = 0
    else:
        start_unit = df.at[i-diff2, 'unit']
        end_unit = df.at[i, 'unit']
        if start_unit == end_unit:
            yes = max(df.iloc[i - diff2 : i]['kruskal2'].to_numpy()) < 5
            df.at[i, 'res'] = 100 if yes else 0
        else:
            df.at[i, 'res'] = 0

exp2 = []
real2 = []
for i, row in df.iterrows():
    if row['res'] == 100:
        exp2.append(row['expected'])
        real2.append(row['forecasted'])

print('CNN total rmse after Kruskal-Wallis ' + str(rmse().calculate(real2, exp2)))
print('CNN total mae after Kruskal-Wallis  ' + str(mae().calculate(real2, exp2)))

plt.plot(output_data)
plt.plot(unitsmagnified)
plt.plot(avg_forecast)
plt.plot(df['res'].to_numpy())
plt.plot(df['kruskal2'].to_numpy(), color = 'pink')
plt.plot(df['kruskal'].to_numpy(), color='yellow')

#plt.plot(avg_forecast2, color = 'green')
plt.show()

