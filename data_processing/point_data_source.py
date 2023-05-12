from typing import List
from .data_source import data_source
from utils.point import point

import pandas as pd
import pickle
from os.path import exists


class point_data_source:
    def get_data(self, filename, window_size) -> List[point]:
        fn = str.replace(filename, '/', '_')
        pickle_filename = f'.local/cache/all_input_{fn}.pickle'
        if exists(pickle_filename):
            print('found all input cache')
            with open(pickle_filename, 'rb') as handle:
                return pickle.load(handle)
       
        raw_records = self.get_raw_records(filename)
        raw_records = self.normalize(raw_records)
        
        output = []
        
        gb = raw_records.groupby('unit')
        for unit, group in gb:
            group_df = pd.DataFrame(group)
            group_df = group_df.reset_index(drop=True)
            max_time = group_df['time'].max()
            
            #print (f'Unit: {unit}, len: {group_df.shape}, max time: {max_time}')

            if group_df.shape[0] >= window_size:
                for i in range(0, group_df.shape[0] - window_size + 1):

                    input_val = pd.DataFrame(group_df[i:i + window_size])
                    input_val = input_val.drop('time', axis='columns')
                    input_val = input_val.drop('unit', axis='columns')

                    output_val = max_time - group_df.loc[i + window_size - 1, 'time']

                    pnt = point(unit = unit, input = input_val.to_numpy(), training_output = output_val)
                    
                    
                    output.append(pnt)

        with open(pickle_filename, 'wb') as handle:
            pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(f'raw: {len(raw_records)}; output:{len(output)}; diff = {len(raw_records) - len(output)}')
        return output
    
    def get_raw_records(self, filename):
        nms =  ['unit', 'time'] + ['s' + str(i) for i in range(1, 25)]
        df =  pd.read_csv(filename, sep = ' ', header = None, names= nms, index_col=False)
        df = df.drop('s3', axis='columns')
        df = df.drop('s4', axis='columns')
        df = df.drop('s8', axis='columns')
        df = df.drop('s9', axis='columns')
        df = df.drop('s13', axis='columns')
        df = df.drop('s19', axis='columns')
        df = df.drop('s21', axis='columns')
        df = df.drop('s22', axis='columns')

        return df
    
    def normalize(self, df: pd.DataFrame):
        result = df.copy()
        for feature_name in df.columns:
            if feature_name.startswith('s'):
                max_value = df[feature_name].max()
                min_value = df[feature_name].min()
                result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
        return result
