#!/usr/bin/env python3
"""a class Normal that represents a normal distribution"""


class Normal:
    """a class Normal that represents a normal distribution"""
    pi = 3.1415926536
    e = 2.7182818285

    def __init__(self, data=None, mean=0., stddev=1.):

        self.data = data
        if data is None:
            if stddev <= 0:
                raise ValueError('stddev must be a positive value')
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.mean = sum(data) / len(data)
            # std_dev = \
            #  pow(sum([(x-self.mean)**2 for x in data])/len(data), 0.5)
            # self.stddev = std_dev
            s = 0
            for x in data:
                s += pow(x-self.mean, 2)
            self.stddev = pow(s/len(data), 0.5)

    def z_score(self, x):
        """z-score"""
        return (x-self.mean)/self.stddev

    def x_value(self, z):
        """x value"""
        return (z*self.stddev) + self.mean

    def pdf(self, x):
        """Probability density function"""
        variance = self.stddev**2
        return (1 / (2 * self.pi * variance**2)**0.5) * self.e ** (-((x-self.mean)**2)/ (2* variance**2))
