from pandas import DataFrame

from data_processing.configurable.stages.pre_point_conversion.base_pre_point_conversion_stage import base_pre_point_conversion_stage


class normalize(base_pre_point_conversion_stage):
    def apply_to(self, df: DataFrame) -> DataFrame:
        result = df.copy()
        for feature_name in df.columns:
            if feature_name.startswith('s'):
                max_value = df[feature_name].max()
                min_value = df[feature_name].min()
                result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
        return result