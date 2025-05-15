from random import random


def order_crossover(parent1, parent2):

    size = len(parent1)
    a, b = sorted(random.sample(range(size), 2))

    hole = set(parent1[a:b+1])
    child = [None] * size

    child[a:b] = parent1[a:b]

    pos = (b+1)%size
    for city in parent2:
        if city not in hole:
            while child[pos] is not None:
                pos = (pos+1) % size
            child[pos] = city
    return child
