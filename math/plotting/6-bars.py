#!/usr/bin/env python3
"""to plot a stacked bar graph"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """to plot a stacked bar graph"""
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4,3))
    people = ['Farrah', 'Fred', 'Felicia']

    apples = fruit[0]
    bananas = fruit[1]
    oranges = fruit[2]
    peaches = fruit[3]

    plt.bar(people, apples, width=0.5, color='red', label='Apples')
    plt.bar(people, bananas, width=0.5, color='yellow', bottom=apples, label='Bananas')
    plt.bar(people, oranges, width=0.5, color='#ff8000', bottom=apples+bananas, label='Oranges')
    plt.bar(people, peaches, width=0.5, color='#ffe5b4', bottom=apples+bananas+oranges, label='Peaches')

    plt.ylabel("Quantity of Fruit")
    plt.yticks(np.arange(0, 81, 10))
    plt.title("Number of Fruit per Person")
    plt.legend()

    plt.show()
