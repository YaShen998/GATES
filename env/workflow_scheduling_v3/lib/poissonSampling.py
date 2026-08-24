import random

import numpy as np

# Reference: https://towardsdatascience.com/the-poisson-process-everything-you-need-to-know-322aa0ab9e9a


def sample_poisson(rate, time):
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
