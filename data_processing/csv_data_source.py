import pandas as pd

from .data_source import data_source

class csv_data_source(data_source):
    def get_data(self, filename, separator):
        
        df =  pd.read_csv(filename, sep = separator, index_col=False)
        
        renaming_rules = {}
        current_index = 1
        for c in df.columns:
            if  c != 'unit' and c != 'time':
                renaming_rules[c] = f's{current_index}'
                current_index += 1
        
        df.rename(renaming_rules, axis = 'columns', inplace = True)

        return df
