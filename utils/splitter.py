from typing import List

from utils.point import point

#=======================================================================================
# splitter
#=======================================================================================
# Класс-обертка для применения кросс-валидации. Многократно разбивает данные согласно 
# введенным параметрам и выполняет переданную функцию для каждого набора данных.
# 
# Входные данные  
#
# Визуализация принципа разбития данных для каждой итерации приведена ниже. 
# Решетки попадают в короткий набор данных, называемый short
# Прочерки попадают в длинный набор данных, называемый long
#---------------------------------------------------------------------------------------
#  ####------------
#  ----####--------
#  --------####----
#  ------------####   
#---------------------------------------------------------------------------------------
# Функции:
# 
# run - выполнить прогон кросс-валидации на входных данных
#   func - функция, которую нужно выполнить на наборе данных. 
#       Сигнатура функции: func(long, short)
#   data - данные для разбития. Массив точек - классов типа point
#   num_of_partitions - на сколько частей разбивать данные. Также, количество итераций.
#   exclusions - список юнитов, которые нужно исключить из данных и которые не должны 
#       участвовать в процессе кросс-валидации
#
# Остальные функции - внутренние
#=======================================================================================
class splitter:
 
    def run(self, func, data, num_of_partitions, exclusions):
        data_wo_exclusions = self.remove_exclusions(data, exclusions)
        
        parts = self.split(data_wo_exclusions, num_of_partitions)
        #print('+=+=+=+')
        #print(range(0, len(parts))
        #print('+=+=+=+')

        output = []
        for i in range(0, len(parts)):
            short = parts[i]
            long = self.join_df([parts[j] for j in range(0, len(parts)) if j != i])
            
            f_o = func(long, short)
            output.append(f_o)
        return output

    def remove_exclusions(self, data: List[point], exclusions: List[int]) -> List[point]:
        return list(filter(lambda x : x.unit not in exclusions, data))

    
    def split(self, data: List[point], num_of_partitions: int) -> List[List[point]]: 
        all_unit_numbers = list(set(map(lambda x : x.unit, data)))
        
        if(len(all_unit_numbers) % num_of_partitions != 0):
            print('total units' + str(len(all_unit_numbers)))
            print('num of partitions' + str(num_of_partitions))
            raise ValueError('Cant split data in equal parts')

        partition_size = int(len(all_unit_numbers) / num_of_partitions)
        
        output = []
        for i in range(0, len(all_unit_numbers), partition_size):
            partition_units = all_unit_numbers[i:i+partition_size]
            partition = list(filter(lambda x: x.unit in partition_units, data))
            output.append(partition)
        return output


    def join_df(self, dfs: List[List]) -> list:
        result = dfs[0].copy()
        for i in range(1, len(dfs)):
            result.extend(dfs[i])
        return result

