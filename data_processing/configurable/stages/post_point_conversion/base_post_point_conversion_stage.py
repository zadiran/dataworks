from typing import List

from utils import point

class base_post_point_conversion_stage:
    def apply_to(self, data: List[point]) -> List[point]:
        pass