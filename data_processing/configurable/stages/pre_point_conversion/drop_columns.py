from typing import List
from pandas import DataFrame
from data_processing.configurable.stages.pre_point_conversion.base_pre_point_conversion_stage import base_pre_point_conversion_stage


class drop_columns(base_pre_point_conversion_stage):
    def __init__(self, columns_to_drop: List[str]):
        self.columns_to_drop = columns_to_drop

    def apply_to(self, df: DataFrame) -> DataFrame:
        result = df.copy()
        
        for col in self.columns_to_drop:
            result = result.drop(col, axis = 'columns')

        return result