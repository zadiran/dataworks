from typing import List
from pandas import DataFrame

from utils import point

class base_point_conversion_stage:
    def apply_to(self, df: DataFrame) -> List[point]:
        pass