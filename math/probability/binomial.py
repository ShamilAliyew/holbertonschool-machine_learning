#!/usr/bin/env python3
"""a class Binomial that represents a binomial distribution"""


class Binomial():
    """a class Binomial that represents a binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        if (data is None):
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p >= 1 or p <= 0:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)
            std_dev = (sum([(x-mean)**2 for x in data]) / len(data))**0.5
            self.p = 1 - (std_dev**2)/mean
            self.n = int(round(mean/self.p))
            self.p = float(mean/self.n)
