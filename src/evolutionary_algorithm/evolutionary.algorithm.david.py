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

def plot_convergence(best_lengths):
    """
    Zeichnet die beste Tour-Länge in jeder Generation.
    best_lengths: Liste der minimalen Längen pro Generation.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(best_lengths, linewidth=2)
    plt.title("GA-Konvergenzverlauf")
    plt.xlabel("Generation")
    plt.ylabel("Beste Tourlänge")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_tour(tour, city_coords):
    """
    Zeichnet die Städte und verbindet sie in der Reihenfolge der Tour.
    tour: Liste von Stadt-Indizes in Besuchsreihenfolge
    city_coords: Nx2-Array mit den Koordinaten
    """
    # Pfad schließen, indem Startstadt ans Ende gehängt wird
    path = tour + [tour[0]]
    xs = [city_coords[i][0] for i in path]
    ys = [city_coords[i][1] for i in path]

    plt.figure(figsize=(6, 6))
    plt.plot(xs, ys, marker='o', linestyle='-')
    for idx in tour:
        x, y = city_coords[idx]
        plt.text(x, y, str(idx), fontsize=9, ha='right')
    plt.title("Beste gefundene Tour")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


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
    children = []

    start = random.randint(0, len(parent1) - 1)
    finish = random.randint(start, len(parent1))
    sub_path_parent1 = parent1[start:finish]

    remaining_path_parent2 = [item for item in parent2 if item not in sub_path_parent1]

    for i in range(len(parent1)):
        if start <= i < finish:
            children.append(sub_path_parent1.pop(0))
        else:
            children.append(remaining_path_parent2.pop(0))

    return children

def mutate(generation, mutation_rate):
    new_generation = [] # neue mutierte generation
    for path in generation:
        new_path = path.copy()
        if random.random() < mutation_rate: # random. random ist ein float value zwischen 0 und 1
            index1, index2 = random.randint(0, len(new_path) - 1), random.randint(0, len(new_path) - 1)
            new_path[index1], new_path[index2] = new_path[index2], new_path[index1]
            new_generation.append(new_path)
    return new_generation



def generate_algorithm(mu, lam, initial_population, mutation_rate, cities_size, city_coords, generations):
    pop = initialize_first_generation(initial_population, cities_size)
    mat = dist_matrix(cities_size, city_coords)



    return



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