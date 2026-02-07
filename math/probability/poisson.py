#!/usr/bin/env python3
"""a class Poisson that represents a poisson distribution"""


class Poisson:
    """a class Poisson that represents a poisson distribution"""
    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        self.data = data
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = sum(data) / len(data)

    def pmf(self, k):
        """Probability density function"""
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0

        first_nom = pow(self.e, -self.lambtha)
        second_nom = pow(self.lambtha, k)

        denominator = 1
        for i in range(1, k + 1):
            denominator *= i

        return (first_nom * second_nom) / denominator

    def cdf(self, k):
        """Cumulative distribution function"""
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0

        cdf_value = 0
        for i in range(k + 1):
            cdf_value += self.pmf(i)

        return cdf_value