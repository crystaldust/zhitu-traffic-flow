
import ctypes

so = ctypes.CDLL('adaptor.so')

cdll = ctypes.cdll.LoadLibrary('adaptor.so')


print('so')