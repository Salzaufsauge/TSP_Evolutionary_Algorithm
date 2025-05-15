from random import random


def generate_random_paths(n):

    random_paths = list[range(n)]
    random.shuffle(random_paths)
    return random_paths