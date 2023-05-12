from typing import List
from data_processing.configurable.stages.post_point_conversion.base_post_point_conversion_stage import base_post_point_conversion_stage
from utils import point


class convert_to_binary_output(base_post_point_conversion_stage):
    def __init__(self, rul_threshold : int):
        self.rul_threshold = rul_threshold

    def apply_to(self, data: List[point]) -> List[point]:
        for pt in data:
            pt.training_output = 1 if pt.training_output < self.rul_threshold else 0

        return data