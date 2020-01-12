import numpy as np
from timer import Timer

timer = Timer()

'''
compares different operations plain python vs numpy
'''

class Comparator:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.dim = len(a)

    def compare_multiply(self):
        print('\nMultiplication')
        c = np.zeros((self.dim, self.dim))
        timer.start()
        c = np.multiply(self.a, self.b)
        print('numpy', timer.stop())

        timer.start()
        for i in range(self.dim):
            for j in range(self.dim):
                for k in range(1):
                    c[i][j] += self.a[i][k] * self.b[k][j]

        print('normal', timer.stop())

    def compare_dot(self):
        print('\nDot Product')
        c = np.zeros((self.dim, self.dim))
        timer.start()
        c = np.dot(self.a, self.b)
        print('numpy', timer.stop())

        timer.start()
        for i in range(self.dim):
            for j in range(self.dim):
                c[i][j] = self.a[i][j] * self.b[i][j]
        print('normal', timer.stop())

    def compare_max(self):
        print('\nMax element')
        timer.start()
        max_ = np.max(a)
        print('numpy', timer.stop())

        timer.start()
        max_ = -1
        for i in range(self.dim):
            for j in range(self.dim):
                max_ = max(self.a[i][j], max_)
        print('normal', timer.stop())

    def compare_average(self):
        print('\nAverage')
        timer.start()
        sum = np.average(a)
        print('numpy', timer.stop())

        timer.start()
        sum = 0
        for i in range(self.dim):
            for j in range(self.dim):
                sum += self.a[i][j]
        sum = sum / (self.dim * self.dim)
        print('normal', timer.stop())


if __name__ == '__main__':
    import inspect

    dim = int(2e3)
    a = np.random.rand(dim, dim).tolist()
    b = np.random.rand(dim, dim).tolist()

    comparator = Comparator(a, b)
    comparator.compare_multiply()
    comparator.compare_dot()
    comparator.compare_max()
    comparator.compare_average()
