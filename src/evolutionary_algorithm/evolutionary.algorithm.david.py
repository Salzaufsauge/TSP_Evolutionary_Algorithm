import math
import random
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
    for tour in population:
        length = tour_length(tour, matrix)
        fitness.append(length)
    return fitness # Für jede Tour in der Population wird die gesamtlänge berechnet

def plus_selection(parents, children, mu, matrix):
    combined = [] # nächste generation besteht aus kinder und eltern
    for tour in parents:
        combined.append(tour)
    for tour in children:
        combined.append(tour)

    lengths = evaluation(combined, matrix) # fitness bzw. länger der tour, wird berechnet

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

def mutate(population, mutation_rate):
    mutated_population = []
    for tour in population:
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

    for gen in range(generations):
        fitness = evaluation(population, matrix)

        # Selektion der Eltern => PArent-Selektion
        paired = list(zip(population, fitness))
        paired.sort(key=lambda x: x[1])  # sortiert nach Tourlänge
        parents = [pair[0] for pair in paired[:mu]]

        # Variation => Crossover mit Mutation
        children = []

        while len(children) < lam:
            p1, p2 = random.sample(parents, 2)
            children.extend(crossover(p1, p2))  # gibt 2 Kinder zurück
        # children = children[:lam]  # evtl. überzählige Kinder abschneiden
        children = mutate(children, mutation_rate)

        # Selektion: beste mu aus Eltern + Kindern
        population = plus_selection(parents, children, mu, matrix)
        # Aktuelle beste Tour bestimmen
        current_best = min(population, key=lambda tour: tour_length(tour, matrix))
        current_best_length = tour_length(current_best, matrix)

        if current_best_length < best_length:
            best_length = current_best_length
            best_tour = current_best.copy()

        history.append(best_length)  # Optional: Für Verlaufsgrafik

        # Optional: Ausgabe pro Generation
        # print(f"Generation {gen + 1}, Beste Länge: {best_length}")

    return best_tour, best_length



def main():
    file = "F:/DEV/PYHTONPROJECTS/TSP/data/eil51.tsp"
    cities, weight_type, city_coords = read_tsp(file)

    mu = 500 # Eltern
    lam = 200 # Nachkommen
    mutation_rate = 0.05
    generations = 500
    initial_population_size = 1500

    best_tour, best_length = generate_algorithm(mu, lam, initial_population_size, mutation_rate,cities, city_coords, generations)
    print("Beste Tour:", best_tour)
    print("Länge:", best_length)

if __name__ == "__main__":
    main()