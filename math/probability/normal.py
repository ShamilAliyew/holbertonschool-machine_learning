#!/usr/bin/env python3
"""a class Normal that represents a normal distribution"""
import math


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
            self.mean = sum(data) / len(data)
            sum = 0
            for x in data:
                sum += pow(x-mean, 2)
            self.stddev = math.sqrt(sum/len(data))
