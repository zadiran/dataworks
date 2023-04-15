from utils.splitter import splitter
from data_processing.test_data_source import test_data_source

def func(long, short):
    print('long')
    print(long)
    print('short')
    print(short)

sp = splitter(test_data_source, 4, [1,2])
sp.run(func)
