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
            std_dev =\
                float((sum([(x-mean)**2 for x in data]) / len(data))**0.5)
            self.p = 1 - (std_dev**2)/mean
            self.n = int(round(mean/self.p))
            self.p = float(mean/self.n)

    def pmf(self, k):
        """Probability mass function"""
        if not isinstance(k, int):
            k = int(k)
        if k is None or k < 0 or k > self.n:
            return 0
        combination = (Binomial.faktorial(self.n) /
                       (Binomial.faktorial(k) *
                        Binomial.faktorial(self.n - k)))
        return combination * (self.p**k) * (1-self.p)**(self.n-k)

    def faktorial(num):
        """faktorial function"""
        f = 1
        for i in range(1, num+1):
            f *= i
        return f

    def cdf(self, k):
        """Cumulative distribution function"""
        if not isinstance(k, int):
            k = int(k)
        if k is None or k < 0 or k > self.n:
            return 0
        cdf_value = 0
        for i in range(k+1):
            cdf_value += self.pmf(i)
        return cdf_value
