from random import random

from src.evolutionary_algorithm.order_crossover import order_crossover
from src.evolutionary_algorithm.swap_mutation import swap_mutation
from src.evolutionary_algorithm.tour_lenght import tour_lenght


def evolve(population, dist_matrix, mu, lam, mautation_rate, generations):

    best_per_generation = []
    for generation in range(generations):
        offspring = []
        for _ in range(lam):
            parent1, parent2 = random.sample(population, 2)
            child = order_crossover(parent1, parent2)
            child = swap_mutation(child, mautation_rate)
            offspring.append(child)
    combined = population + offspring
    combined.sort(key=lambda random_paths: tour_lenght(random_paths, dist_matrix))
    population = combined[:mu]
    best_per_generation.append(tour_lenght(population[0], dist_matrix))

    return population[0], best_per_generation