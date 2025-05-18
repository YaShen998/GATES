import random

import numpy as np


# The Poisson Process: Everything you need to know
# https://towardsdatascience.com/the-poisson-process-everything-you-need-to-know-322aa0ab9e9a

# rate: number of arrivals per second, time: in how many seconds
def sample_poisson(rate, time):  # Sample from a possion process
    pos_array = []
    current = 0
    while True:
        pos = -(np.log(1 - random.random())) / rate
        current += pos
        if current < time:
            pos_array.append(current)
        else:
            return pos_array

def one_sample_poisson(rate, startTime):
    current = startTime
    while True:
        pos = -(np.log(1 - random.random())) / rate
        current += pos
        return current

# sample a fixed number of data from the distribution
def num_sample_poisson(rate, startTime, num):
    pos_array = []
    current = startTime
    while True:
        pos = -(np.log(1 - random.random())) / rate
        current += pos
        if len(pos_array) < num:
            pos_array.append(current)
        else:
            return pos_array

# print(sample_possion(5/3600, 3600))
# x = np.random.poisson(0.1, 100)
# print(x, sum(x))