from typing import List
from pandas import DataFrame

from data_processing.configurable.stages.point_conversion.base_point_conversion_stage import base_point_conversion_stage
from data_processing.configurable.stages.post_point_conversion.base_post_point_conversion_stage import base_post_point_conversion_stage
from data_processing.configurable.stages.pre_point_conversion.base_pre_point_conversion_stage import base_pre_point_conversion_stage
from utils import point


class configurable_data_manipulator:
    def __init__(self):
        self.pre_point_conversion_stages: List[base_pre_point_conversion_stage] = []
        self.point_conversion_stage: base_point_conversion_stage = None
        self.post_point_conversion_stages: List[base_post_point_conversion_stage] = []

    # Methods for pre point conversion stages
    def add_pre_point_conversion_stage(self, stage : base_pre_point_conversion_stage):
        self.pre_point_conversion_stages.append(stage)

    def clear_pre_point_conversion_stages(self):
        self.pre_point_conversion_stages = []

    # Methods for point conversion stage
    def set_point_conversion_stage(self, stage : base_point_conversion_stage):
        self.point_conversion_stage = stage

    def clear_point_conversion_stage(self):
        self.point_conversion_stage = None

    # Methods for post point conversion stages
    def add_post_point_conversion_stage(self, stage : base_post_point_conversion_stage):
        self.post_point_conversion_stages.append(stage)

    def clear_post_point_conversion_stages(self):
        self.pre_post_conversion_stages = []

    # Main method
    def get_processed_data(self, raw_data: DataFrame) -> List[point]:
        if self.point_conversion_stage is None:
            raise RuntimeError('Point conversion stage is not configured')
        
        df = raw_data.copy(deep = True)

        for stage in self.pre_point_conversion_stages:
            df = stage.apply_to(df)

        points = self.point_conversion_stage.apply_to(df)

        for stage in self.post_point_conversion_stages:
            points = stage.apply_to(points)

        return points
        