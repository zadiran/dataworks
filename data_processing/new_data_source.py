from msilib import datasizemask
from typing import List
from data_source import data_source
from utilities.point import point

class new_data_source(data_source):
    def __init__(self, file_path: str, window_size: int):
        self.file_path = file_path
        self.window_size = window_size
    
    def get_data(self) -> List[point]:
        pass