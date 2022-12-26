import pandas as pd

class nasa_data_source:
    def __init__(self, filepath_) -> None:
        self.filepath = filepath_
        self.df = None

    def get_data(self) -> pd.DataFrame:
        if  self.df is None:
            nms =  ['unit', 'time'] + ['s' + str(i) for i in range(1, 27)]
            self.df = pd.read_csv(self.filepath, sep = ' ', header = None, names= nms, index_col=False)
        return self.df