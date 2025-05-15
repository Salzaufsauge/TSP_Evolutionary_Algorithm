from random import random


def swap_mutation(random_paths, mutation_rate):
    if random.random() < mutation_rate:
        a, b = random.sample(range(len(random_paths)), 2)
        random_paths[a], random_paths[b] = random_paths[b], random_paths[a]
    return random_paths