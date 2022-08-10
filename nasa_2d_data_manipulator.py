from itertools import cycle
import numpy as np
import pandas as pd
import pickle
from os.path import exists

from nasa_data_source import nasa_data_source


class nasa_2d_data_manipulator:
    def __init__(self, data_source: nasa_data_source) -> None:
        self.data_source = data_source
        self.cv_start : int
        self.cv_end : int
        self.window_size : int = 20

    def set_cv_range(self, start, end):
        self.cv_start = start
        self.cv_end = end

    def get_param_cols(self):
        arr = ['s' + str(i) for i in range(1, 25)]
        arr.remove('s3')
        arr.remove('s4')
        arr.remove('s8')
        arr.remove('s9')
        arr.remove('s13')
        arr.remove('s19')
        arr.remove('s21')
        arr.remove('s22')
        return arr

    def get_exclusion_range(self):
        return [96, 97, 98, 99, 100]

    def normalize(self, df):
        result = df.copy()
        for feature_name in self.get_param_cols():
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
        return result

    def get_marked_dataset(self):
        columns = ['unit', 'time'] + ['s' + str(i) for i in range(1, 25)]
        df = self.data_source.get_data()[columns]
        df = df.drop('s3',1)
        df = df.drop('s4',1)
        df = df.drop('s8',1)
        df = df.drop('s9',1)
        df = df.drop('s13',1)
        df = df.drop('s19',1)
        df = df.drop('s21',1)
        df = df.drop('s22',1)

        df = self.normalize(df)
        return df

    def get_training_input(self):
        pickle_filename = 'training_input_' + str(self.cv_start) + '_' + str(self.cv_end) + '.pickle'
        if exists(pickle_filename):
            print('found training input cache for range ' + str(self.cv_start) + '-' + str(self.cv_end))
            with open(pickle_filename, 'rb') as handle:
                return pickle.load(handle)

        result = []
        cv_data = self.get_cv_training_input()
        cv_data = cv_data.reset_index()
        for i in range(0, cv_data.shape[0] - self.window_size):
            cycle_start = cv_data.at[i, 'unit']
            cycle_end = cv_data.at[i + self.window_size - 1, 'unit']
            if cycle_start == cycle_end:
                dfa = cv_data.iloc[i : i + self.window_size]
                dfa = dfa.drop('unit', 1)
                dfa = dfa.drop('time', 1)
                result.append(dfa[self.get_param_cols()].to_numpy())
        
        arr = np.array(result)
        with open(pickle_filename, 'wb') as handle:
            pickle.dump(arr, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        return arr


    def get_training_output(self):
        pickle_filename = 'training_output_' + str(self.cv_start) + '_' + str(self.cv_end) + '.pickle'
        if exists(pickle_filename):
            print('found training output cache for range ' + str(self.cv_start) + '-' + str(self.cv_end))
            with open(pickle_filename, 'rb') as handle:
                return pickle.load(handle)

        cv_data = self.get_cv_training_output()
        prox = self.get_proximity_for_arr(cv_data)
        result = prox

        arr = np.array(result)
        with open(pickle_filename, 'wb') as handle:
            pickle.dump(arr, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        return arr
    
    def get_testing_input(self):
        pickle_filename = 'testing_input_' + str(self.cv_start) + '_' + str(self.cv_end) + '.pickle'
        if exists(pickle_filename):
            print('found testing input cache for range ' + str(self.cv_start) + '-' + str(self.cv_end))
            with open(pickle_filename, 'rb') as handle:
                return pickle.load(handle)

        result = []
        cv_data = self.get_cv_testing_input()
        cv_data = cv_data.reset_index()
        maxiterindex = cv_data.shape[0] - self.window_size
        for i in range(0, maxiterindex):
            cycle_start = cv_data.at[i, 'unit']
            cycle_end = cv_data.at[i + self.window_size - 1, 'unit']
            if cycle_start == cycle_end and (i + 1 == maxiterindex or cv_data.at[i + self.window_size, 'unit'] == cycle_start):
                dfa = cv_data.iloc[i : i + self.window_size]
                dfa = dfa.drop('unit', 1)
                dfa = dfa.drop('time', 1)
                result.append(dfa[self.get_param_cols()].to_numpy())
        
        arr = np.array(result)
        with open(pickle_filename, 'wb') as handle:
            pickle.dump(arr, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return arr

    def get_testing_output(self):
        pickle_filename = 'testing_output_' + str(self.cv_start) + '_' + str(self.cv_end) + '.pickle'
        if exists(pickle_filename):
            print('found testing output cache for range ' + str(self.cv_start) + '-' + str(self.cv_end))
            with open(pickle_filename, 'rb') as handle:
                return pickle.load(handle)

        cv_data = self.get_cv_testing_output()
        prox = self.get_proximity_for_arr(cv_data)
        result = prox#[self.window_size:]

        arr = np.array(result)
        with open(pickle_filename, 'wb') as handle:
            pickle.dump(arr, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return arr

    def get_cv_training_input(self):
        df = self.get_marked_dataset()
        df_tr = df.loc[~df['unit'].isin(list(range(self.cv_start, self.cv_end)) + list(self.get_exclusion_range()))]
        return df_tr

    def get_cv_testing_input(self):
        df = self.get_marked_dataset()
        df_ts = df.loc[df['unit'].isin(list(range(self.cv_start, self.cv_end)))]
        return df_ts

    def get_clean_training_input(self):
        ti = self.get_cv_training_input()
        ti = ti.drop('unit', 1)
        ti = ti.drop('time', 1)
        return ti

    def get_clean_testing_input(self):
        ti = self.get_cv_testing_input()
        ti = ti.drop('unit', 1)
        ti = ti.drop('time', 1)
        return ti
    
    def get_units(self):
        df = self.get_marked_dataset()[['unit', 'time']]
        df = df.loc[df['unit'].isin(list(range(self.cv_start, self.cv_end)))]
        
        for i, row in df.iterrows():
            row['time'] -= 20

        dfsorted = df.loc[df['time'] > 0]    
        
        return dfsorted['unit'].to_numpy()

        
    def get_cv_training_output(self):
        df = self.get_marked_dataset()[['unit', 'time']]
        df = df.loc[~df['unit'].isin(list(range(self.cv_start, self.cv_end)) + list(self.get_exclusion_range()))]
        time = df['time'].to_numpy()
        
        output = []
        cnt = 0
        for i in range(0, len(time)):
            if time[i] > time[i-1]:
                cnt += 1
            else: 
                output += [0 for j in range(0, cnt - self.window_size + 1)]
                output.append(1)
                cnt = 0

        output += [0 for j in range(0, cnt)]
        output.append(1)
        return output

    def get_cv_testing_output(self):
        df = self.get_marked_dataset()[['unit', 'time']]
        df = df.loc[df['unit'].isin(list(range(self.cv_start, self.cv_end)))]
        time = df['time'].to_numpy()
        
        print('====================================')
        print('len time b4: ' + str(len(time)))

        time = [t - 20 for t in time]
        timeclr = []
        for t in time:
            if t > 0:
                timeclr.append(t) 
        time = timeclr
        print('len time after: ' + str(len(time)))
        print('====================================')

        output = []
        cnt = 0
        for i in range(1, len(time)):
            if time[i] > time[i-1]:
                cnt += 1
            else: 
                output += [0 for j in range(0, cnt)]
                output.append(1)
                cnt = 0

        output += [0 for j in range(0, cnt)]
        output.append(1)

        print('len output data: ' + str(len(output)))
        print('====================================')

        return output
    
    def get_proximity_for_arr(self, arr):
        res = []
        last_failure = -1
        for i in range(0, len(arr)):
            if arr[i] == 0:
                continue
            else:
                cnt = i - last_failure 
                res += list(reversed(range(0, cnt)))
                last_failure = i 

        return res

    def get_test_proximity(self):
        arr = self.get_testing_output()
        res = []
        last_failure = -1
        for i in range(0, len(arr)):
            if arr[i] == 0:
                continue
            else:
                cnt = i - last_failure 
                res += list(reversed(range(0, cnt)))
                last_failure = i 

        return res

    def get_logical_proximity(self, array, distance):
        proximity = self.get_proximity_for_arr(array)
        output = []
        for i in proximity:
            if i <= distance:
                output.append(1)
            else:
                output.append(0)
        return output

    def get_test_logical_proximity(self, distance):
        proximity = self.get_test_proximity()
        output = []
        for i in proximity:
            if i <= distance:
                output.append(1)
            else:
                output.append(0)
        return output