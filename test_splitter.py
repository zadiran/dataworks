from point import point
from splitter import splitter
from test_data_source import test_data_source

def func(long, short):
    print('================================')
    print('long')
    print(long)
    print('--------------------------------')
    print('short')
    print(short)

sp = splitter(test_data_source, 4, [1,2])
sp.run(func)
