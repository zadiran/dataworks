from .nasa_data_source import nasa_data_source

class nasa_data_manipulator:
    def __init__(self, data_source: nasa_data_source) -> None:
        self.data_source = data_source

    def set_cv_range(self, start, end):
        self.cv_start = start
        self.cv_end = end

    def get_exclusion_range(self):
        return [96, 97, 98, 99, 100]
    
    def get_full_dataset(self):
        columns = ['unit', 'time'] + ['s' + str(i) for i in range(1, 25)]
        df = df = self.data_source.get_data()[columns]
        return df
    
    def get_cv_training_input(self):
        df = self.get_full_dataset()
        df_tr = df.loc[~df['unit'].isin(list(range(self.cv_start, self.cv_end)) + self.get_exclusion_range())]
        return df_tr

    def get_clean_validation_input(self):
        df = self.get_full_dataset()
        df_tr = df.loc[df['unit'].isin(self.get_exclusion_range())]
        df_tr = df_tr.drop('unit', axis='columns')
        df_tr = df_tr.drop('time', axis='columns')
        return df_tr.to_numpy()

    def get_clean_validation_output(self):
        df = self.get_full_dataset()[['unit', 'time']]
        df = df.loc[df['unit'].isin(self.get_exclusion_range())]
        time = df['time'].to_numpy()
        
        output = []
        cnt = 0
        for i in range(1, len(time)):
            if time[i] > time[i-1]:
                cnt += 1
            else: 
                output += [0 for j in range(0, cnt)]
                output.append(1)
                cnt = 0

        output += [0 for j in range(0, cnt)]
        output.append(1)
        return self.get_proximity_for_arr(output)

    def get_cv_testing_input(self):
        df = self.get_full_dataset()
        df_ts = df.loc[df['unit'].isin(list(range(self.cv_start, self.cv_end)))]
        return df_ts

    def get_clean_training_input(self):
        ti = self.get_cv_training_input()
        ti = ti.drop('unit', axis='columns')
        ti = ti.drop('time', axis='columns')
        return ti

    def get_clean_testing_input(self):
        ti = self.get_cv_testing_input()
        ti = ti.drop('unit', axis='columns')
        ti = ti.drop('time', axis='columns')
        return ti
    
    def get_cv_training_output(self):
        df = self.get_full_dataset()[['unit', 'time']]
        df = df.loc[~df['unit'].isin(list(range(self.cv_start, self.cv_end)) + self.get_exclusion_range())]
        time = df['time'].to_numpy()
        
        output = []
        cnt = 0
        for i in range(1, len(time)):
            if time[i] > time[i-1]:
                cnt += 1
            else: 
                output += [0 for j in range(0, cnt)]
                output.append(1)
                cnt = 0

        output += [0 for j in range(0, cnt)]
        output.append(1)
        return output

    def get_cv_testing_output(self):
        df = self.get_full_dataset()[['unit', 'time']]
        df = df.loc[df['unit'].isin(list(range(self.cv_start, self.cv_end)))]
        time = df['time'].to_numpy()
        
        output = []
        cnt = 0
        for i in range(1, len(time)):
            if time[i] > time[i-1]:
                cnt += 1
            else: 
                output += [0 for j in range(0, cnt)]
                output.append(1)
                cnt = 0

        output += [0 for j in range(0, cnt)]
        output.append(1)
        return output
    
    def get_proximity_for_arr(self, arr):
        res = []
        last_failure = -1
        for i in range(0, len(arr)):
            if arr[i] == 0:
                continue
            else:
                cnt = i - last_failure 
                res += list(reversed(range(0, cnt)))
                last_failure = i 

        return res
