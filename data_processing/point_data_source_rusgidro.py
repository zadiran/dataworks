from typing import List
from data_processing import csv_data_source
from utils.point import point

import pandas as pd
import pickle
from os.path import exists


class point_data_source_rusgidro:
    def get_data(self, filename, window_size) -> List[point]:
               
        raw_records = csv_data_source().get_data(filename, ';')
        raw_records = raw_records.drop('s8', 'columns')
        #raw_records = self.normalize(raw_records)
        
        output : List[point] = []
        
        gb = raw_records.groupby('unit')
        for unit, group in gb:
            group_df = pd.DataFrame(group)
            group_df = group_df.reset_index(drop=True)
            max_time = group_df['time'].max()
            
            print (f'Unit: {unit}, len: {group_df.shape}, max time: {max_time}')

            if group_df.shape[0] >= window_size:
                for i in range(0, group_df.shape[0] - window_size + 1):

                    input_val = pd.DataFrame(group_df[i:i + window_size])
                    input_val = input_val.drop('time', 'columns')
                    input_val = input_val.drop('unit', 'columns')

                    output_val = max_time - group_df.loc[i + window_size - 1, 'time']
                    input_val_1 = []
                    for x in input_val.values:
                        input_val_1 = input_val_1 + list(x)
                    pnt = point(unit = unit, input = input_val_1, training_output = output_val)
                    #pnt = point(unit = unit, input = input_val, output = output_val)
                    
                    
                    output.append(pnt)

        print(f'raw: {len(raw_records)}; output:{len(output)}; diff = {len(raw_records) - len(output)}')
        print(output[0].input)
        return output
    
    def normalize(self, df: pd.DataFrame):
        result = df.copy()
        for feature_name in df.columns:
            if feature_name.startswith('s'):
                max_value = df[feature_name].max()
                min_value = df[feature_name].min()
                print(min_value)
                result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
        return result
