from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score

import matplotlib.pyplot as plt

from data_processing.nasa_data_source import nasa_data_source
from data_processing.nasa_data_manipulator import nasa_data_manipulator
from models.baseline_binary_forecast_model import baseline_binary_forecast_model

from utils.splitter import splitter

ds = nasa_data_source('data/train_FD001.txt')
dm = nasa_data_manipulator(ds)

output = []
expected = []


for i in range(0,5):
    fm = baseline_binary_forecast_model(dm)
    
    dm.set_cv_range(i * 20 + 1, (i+1) * 20 + 1)
    fm.fit()

    for row in dm.get_clean_testing_input().to_numpy():
        output.append(fm.predict(row)[0])

    expected +=  dm.get_logical_proximity(dm.get_cv_testing_output(), 10)


print('ROC/AUC: ' + str(roc_auc_score(output, expected)))
print('F1: ' + str(f1_score(output, expected)))
print('Accuracy: ' + str(accuracy_score(output, expected)))
print('Precision: ' + str(precision_score(output, expected)))

plt.plot(expected[:500], color = 'orange')
plt.scatter(range(0,500), output[:500])
plt.show()