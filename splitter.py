from typing import List
import pandas as pd

from data_source import data_source
from point import point


class splitter:
    def __init__(self, data_source : data_source, num_of_partitions : int, exclusions : List[int]):
        self.data_source = data_source
        self.num_of_partitions = num_of_partitions
        self.exclusions = exclusions

    def run(self, func, _data):
        data = self.remove_exclusions(_data)
        
        parts = self.split(data)

        output = []
        for i in range(0, len(parts)):
            if i > 0:
                continue
            short = parts[i]
            long = self.join_df([parts[j] for j in range(0, len(parts)) if j != i])
            
            f_o = func(long, short)
            output.append(f_o)
        return output

    def remove_exclusions(self, data: List[point]) -> List[point]:
        return list(filter(lambda x : x.unit not in self.exclusions, data))

    
    def split(self, data: List[point]) -> List[List[point]]: 
        all_unit_numbers = list(set(map(lambda x : x.unit, data)))
        
        if(len(all_unit_numbers) % self.num_of_partitions != 0):
            raise ValueError('Cant split data in equal parts')

        partition_size = int(len(all_unit_numbers) / self.num_of_partitions)
        
        output = []
        for i in range(0, len(all_unit_numbers), partition_size):
            partition_units = all_unit_numbers[i:i+partition_size]
            partition = list(filter(lambda x: x.unit in partition_units, data))
            output.append(partition)
        return output


    def join_df(self, dfs: List[List]) -> list:
        result = dfs[0].copy()
        for i in range(1, len(dfs)):
            result.extend(dfs[i])
        return result

