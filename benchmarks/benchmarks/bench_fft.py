from .common import Benchmark

import numpy as np

class FFT(Benchmark):
    params = [2**x for x in range(6,26)]
    param_names = ['size']

    def setup(self,size):
        self.x1=np.ones(size,dtype=np.complex)
        self.x1+=1j
        self.x2=np.ones((size,4),dtype=np.complex)
        self.x2+=1j

    def time_1dfft(self,size):
        np.fft.fft(self.x1)

    def time_1difft(self,size):
        np.fft.ifft(self.x1)

    def time_2dfft(self,size):
        np.fft.fft2(self.x2)

    def time_2difft(self,size):
        np.fft.ifft2(self.x2)
