
from typing import List
import pandas as pd
from data_processing.configurable.stages.point_conversion.base_point_conversion_stage import base_point_conversion_stage
from utils import point


class convert_to_2d_input(base_point_conversion_stage):
     def __init__(self, window_size: int):
         self.window_size = window_size

     def apply_to(self, df: pd.DataFrame) -> List[point]:
        output = []
        
        gb = df.groupby('unit')
        for unit, group in gb:
            group_df = pd.DataFrame(group)
            group_df = group_df.reset_index(drop=True)
            max_time = group_df['time'].max()
            
            #print (f'Unit: {unit}, len: {group_df.shape}, max time: {max_time}')

            if group_df.shape[0] >= self.window_size:
                for i in range(0, group_df.shape[0] - self.window_size + 1):

                    input_val = pd.DataFrame(group_df[i:i + self.window_size])
                    input_val = input_val.drop('time', axis='columns')
                    input_val = input_val.drop('unit', axis='columns')

                    output_val = max_time - group_df.loc[i + self.window_size - 1, 'time']

                    pnt = point(unit = unit, input = input_val.to_numpy(), training_output = output_val)
                    
                    
                    output.append(pnt)

        # with open(pickle_filename, 'wb') as handle:
        #     pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(f'raw: {len(df)}; output:{len(output)}; diff = {len(df) - len(output)}')
        return output