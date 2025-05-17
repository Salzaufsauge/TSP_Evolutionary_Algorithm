import math
import random
import time

import matplotlib.pyplot as plt
import numpy as np

def read_tsp(file):
    with open(file, 'r') as f:
        lines = f.readlines()
        weight_type = None
        found_node_coord_section = False
        cities = None
        city_coords = []
        for line in lines:
            if line.startswith('DIMENSION'):
                cities = int(line.split(':')[1].strip())
                continue
            if line.startswith('EDGE_WEIGHT_TYPE'):
                weight_type = line.split(':')[1].strip()
                continue
            if line.startswith('NODE_COORD_SECTION'):
                found_node_coord_section = True
                continue
            if found_node_coord_section:
                if line.startswith('EOF'):
                    break
                node_id, x, y = line.strip().split()
                city_coords.append((float(x), float(y)))
        return cities, weight_type, np.array(city_coords)

def dist_matrix(cities_size, city_coords):
        dist = np.zeros((cities_size, cities_size), dtype=float)  # Matrix mit [cities] * [cities] Zeilen und Spalten, default ist float64, man kann auch auf int32 wechseln
        for i in range(cities_size):
            xi, yi = city_coords[i]
            for j in range(cities_size):
                xj, yj = city_coords[j]
                dx = xi - xj  # Unterschied in x-Richtung
                dy = yi - yj  # Unterschied in y-Richtung
                dist[i][j] = round(math.sqrt(dx * dx + dy * dy)) # Euklidische Distanz 2D gerundet, aus der verlinkten Doc vom Prof.
        return dist

def initialize_first_generation(population_size, cities_size):
    random_paths = []
    for i in range(population_size):
        random_path = list(range(cities_size))
        random.shuffle(random_path)
        random_paths.append(random_path)
    return random_paths

def tour_length(tour, matrix): # Matrix kann beliebiges Berechnungsschema sein. Erweiterbar auf andere Weight_Types
    distance = 0
    best_tour = None
    best_distance = None

    for i in range(len(tour) - 1):
        distance += matrix[tour[i], tour[i + 1]]
    distance += matrix[tour[-1], tour[0]] # hier wird noch die letzte Tour berechnet, also zur Anfangsstadt
    return distance

def evaluation(population, matrix):
    fitness = []
    best_length = float('inf')
    best_tour = None

    for tour in population:
        length = tour_length(tour, matrix)
        fitness.append(length)
        if length < best_length:
            best_length = length
            best_tour = tour.copy()

    return fitness, best_tour, best_length # Für jede Tour in der Population wird die gesamtlänge berechnet. best_tour und best_length bestimmt.


def parent_selection(population, fitness, mu):
    paired = []
    for i in range(len(population)):
        paired.append([population[i], fitness[i]])

    paired.sort(key=lambda x: x[1])  # Sortiere nach Tourlänge

    parents = []
    for i in range(mu):
        parents.append(paired[i][0])

    return parents

def variation(parents, lam, mutation_rate):
    children = []
    while len(children) < lam:
        p1, p2 = random.sample(parents, 2)
        children.extend(crossover(p1, p2))
    children = mutate(children, mutation_rate)
    return children

def plus_selection(parents, children, mu, matrix):
    combined = [] # nächste generation besteht aus kinder und eltern
    for tour in parents:
        combined.append(tour)
    for tour in children:
        combined.append(tour)

    lengths, _, _ = evaluation(combined, matrix) # fitness bzw. länger der tour, wird berechnet

    paired = [] # (tour, length). könnte man auch mit zip machen
    for i in range(len(combined)):
        paired.append((combined[i], lengths[i]))

    paired.sort(key=lambda pair: pair[1]) # sortiere liste nach pair[1] aufsteigend, bzw. die länge der touren z.B. Tourlänge 90, 100, 200, ...

    next_generation = [] # entsprechend der höhe von mu werden die top individuen genommen bzw. die Individuen mit den besten Touren. Aus Eltern und Kindern
    for i in range(mu):
        next_generation.append(paired[i][0])
    return next_generation


def crossover(parent1, parent2):
    child1 = []
    child2 = []

    start = random.randint(0, len(parent1) - 1)
    finish = random.randint(start, len(parent1))

    sub_path_parent1 = parent1[start:finish] # Child 1
    remaining_path_parent2 = [item for item in parent2 if item not in sub_path_parent1]

    for i in range(len(parent1)):
        if start <= i < finish:
            child1.append(sub_path_parent1.pop(0))
        else:
            child1.append(remaining_path_parent2.pop(0))

    sub_path_parent2 = parent2[start:finish] # Child 2
    remaining_path_parent1 = [item for item in parent1 if item not in sub_path_parent2]

    for i in range(len(parent2)):
        if start <= i < finish:
            child2.append(sub_path_parent2.pop(0))
        else:
            child2.append(remaining_path_parent1.pop(0))

    return [child1, child2]

# def reellwertige_mutation(mu, sigma, mutation_rate):
#     mutated_population = []
#     for individual in mu:
#         new_individual = individual.copy()
#         if random.random() < mutation_rate:
#             noise = np.random.normal(0, 1, size=len(individual))  # N(0,1) für jeden Wert np.random.normal(Erwartungswert, Standardabweichung und länge des invidivuals bzw. bei bier127 = 127)
#             new_individual = new_individual + sigma * noise
#         mutated_population.append(new_individual)
#     return mutated_population

def mutate(lam, mutation_rate):
    mutated_population = []
    for tour in lam:
        new_tour = tour.copy()
        if random.random() < mutation_rate:
            i, j = random.sample(range(len(tour)), 2)
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        mutated_population.append(new_tour)
    return mutated_population

def generate_algorithm(mu, lam, initial_population_size, mutation_rate, cities_size, city_coords, generations):
    best_tour = None
    best_length = float('inf')
    history = []

    # Initiale Population und Distanzmatrix
    population = initialize_first_generation(initial_population_size, cities_size)
    matrix = dist_matrix(cities_size, city_coords)

    # Evaluation der ersten population
    fitness, _, _ = evaluation(population, matrix)


    for gen in range(generations):
        # Selektion der Eltern => Parent-Selektion
        parents = parent_selection(population, fitness, mu)

        # Variation => Crossover mit Mutation
        children = variation(parents, lam, mutation_rate)

        # Evaluation
        fitness, current_best, current_best_length = evaluation(population, matrix)

        # Selektion: beste mu aus Eltern + Kindern
        population = plus_selection(parents, children, mu, matrix)

        # für plot
        if current_best_length < best_length:
            best_length = current_best_length
            best_tour = current_best.copy()

        history.append(best_length)

        print(f"Generation {gen + 1}, Beste Länge: {best_length}")

    return best_tour, best_length, history



def main():
    t0 = time.time()
    file = "F:/DEV/PYHTONPROJECTS/TSP/data/bier127.tsp"
    cities, weight_type, city_coords = read_tsp(file)

    mu = 1000 # Eltern
    lam = 500 # Nachkommen
    mutation_rate = 0.15
    generations = 1000
    initial_population_size = 1000

    best_tour, best_length, history = generate_algorithm(mu, lam, initial_population_size, mutation_rate,cities, city_coords, generations)
    print("Beste Tour:", best_tour)
    print("Länge:", best_length)

    # Plot 1: Verlauf der besten Tourlänge
    plt.plot(history)
    plt.xlabel("Generation")
    plt.ylabel("Tourlänge")
    plt.title("Verlauf der besten Tourlänge")
    plt.grid(True)
    plt.show()

    # Plot 2: Visualisierung der besten Tour
    tour = best_tour + [best_tour[0]]  # Rundtour
    x, y = zip(*[city_coords[i] for i in tour])
    plt.figure()
    plt.plot(x, y, 'o-')
    plt.title("Beste gefundene Tour")
    plt.axis("equal")
    plt.grid(True)
    plt.show()

    t1 = time.time()
    print(t1 - t0)

if __name__ == "__main__":
    main()