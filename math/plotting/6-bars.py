#!/usr/bin/env python3
"""to plot a stacked bar graph"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """to plot a stacked bar graph"""
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4,3))
    plt.figure(figsize=(6.4, 4.8))

    people = ['Farrah', 'Fred', 'Felicia']

    apples = fruit[0]
    bananas = fruit[1]
    oranges = fruit[2]
    peaches = fruit[3]

    plt.bar(people, apples, label="Apples", color='red', width=0.5)
    plt.bar(people, bananas, label="Bananas", color='yellow', width=0.5, bottom=apples)
    plt.bar(people, oranges, label="Oranges", color='#ff8000', width=0.5, bottom=apples+bananas)
    plt.bar(people, peaches, label="Peaches", color='#ffe5b4', width=0.5, bottom=apples+bananas+oranges
            )

    plt.ylabel("Quantity of Fruit")
    plt.yticks(np.arange(0, 81, 10))
    plt.ylim(0, 80)
    plt.title("Number of Fruit per Person")
    plt.legend(['Apples', 'Bananas', 'Oranges', 'Peaches'])

    plt.show()
