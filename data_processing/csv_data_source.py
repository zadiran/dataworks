import pandas as pd

from utils.constants.column_names import unit_col, time_col
from .data_source import data_source

class csv_data_source(data_source):
    def get_data(self, filename, separator):
        
        df =  pd.read_csv(filename, sep = separator, index_col=False)
        
        # check if input file has required columns
        if unit_col not in df.columns or time_col not in df.columns:
            raise ValueError('Required columns are missing from input file')

        renaming_rules = {}
        current_index = 1
        for c in df.columns:
            if  c != unit_col and c != time_col:
                renaming_rules[c] = f's{current_index}'
                current_index += 1
        
        df.rename(renaming_rules, axis = 'columns', inplace = True)

        return df
