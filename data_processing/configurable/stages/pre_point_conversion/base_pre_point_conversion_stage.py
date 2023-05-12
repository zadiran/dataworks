from pandas import DataFrame

class base_pre_point_conversion_stage:
    def apply_to(self, df: DataFrame) -> DataFrame:
        pass