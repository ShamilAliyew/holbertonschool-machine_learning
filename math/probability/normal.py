#!/usr/bin/env python3
"""a class Normal that represents a normal distribution"""


class Normal:
    """a class Normal that represents a normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):

        self.data = data
        if data is None:
            if stddev <= 0:
                raise ValueError('the message stddev must be a positive value')
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.mean = sum(data) / len(data)
            s = 0
            for x in data:
                s += pow(x-mean, 2)
            self.stddev = pow(s/len(data), 0.5)
