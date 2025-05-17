import numpy as np
import random
import matplotlib.pyplot as plt

# --- TSPLIB-Reader (wie im ersten Code) ---
def read_tsplib(filename):
    with open(filename, 'r') as f:
        coords = []
        found = False
        for line in f:
            if 'NODE_COORD_SECTION' in line:
                found = True
                continue
            if found:
                if 'EOF' in line:
                    break
                parts = line.strip().split()
                if len(parts) >= 3:
                    coords.append((float(parts[1]), float(parts[2])))
        return np.array(coords)

# --- Distanzmatrix berechnen (O(1)-Lookup) ---
def distance_matrix(coords):
    n = len(coords)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = np.linalg.norm(coords[i] - coords[j])
    return dist

# --- Tourlänge (Fitness) berechnen ---
def tour_length(tour, dist_matrix):
    return sum(dist_matrix[tour[i], tour[(i+1)%len(tour)]] for i in range(len(tour)))

# --- Individuum erzeugen (Zufallspermutation) ---
def create_individual(n):
    ind = list(range(n))
    random.shuffle(ind)
    return ind

# --- Order Crossover (OX) ---
def order_crossover(p1, p2):
    size = len(p1)
    a, b = sorted(random.sample(range(size), 2))
    hole = set(p1[a:b+1])
    child = [None]*size
    child[a:b+1] = p1[a:b+1]
    pos = (b+1) % size
    for city in p2:
        if city not in hole:
            while child[pos] is not None:
                pos = (pos+1) % size
            child[pos] = city
    return child

# --- Swap-Mutation ---
def swap_mutation(ind, mutation_rate):
    if random.random() < mutation_rate:
        a, b = random.sample(range(len(ind)), 2)
        ind[a], ind[b] = ind[b], ind[a]
    return ind

# --- Erzeuge genau 2 Nachkommen aus 2 Eltern ---
def generate_two_offspring(p1, p2, mutation_rate):
    # Kind 1: p1→p2
    c1 = order_crossover(p1, p2)
    c1 = swap_mutation(c1, mutation_rate)
    # Kind 2: p2→p1
    c2 = order_crossover(p2, p1)
    c2 = swap_mutation(c2, mutation_rate)
    return c1, c2

# --- Evolve-Funktion mit 2-Offspring-Pro-Pair-Strategie ---
def evolve(population, dist_matrix, mu, lam, mutation_rate, generations):
    best_per_gen = []
    for gen in range(generations):
        offspring = []
        # Wir erzeugen lam Kinder, je zwei pro Elternpaar
        pairs = lam // 2
        for _ in range(pairs):
            p1, p2 = random.sample(population, 2)
            c1, c2 = generate_two_offspring(p1, p2, mutation_rate)
            offspring.extend([c1, c2])
        # Falls lam ungerade, noch ein zusätzliches Kind
        if lam % 2 == 1:
            p1, p2 = random.sample(population, 2)
            c1, _ = generate_two_offspring(p1, p2, mutation_rate)
            offspring.append(c1)

        # (μ + λ) Selektion
        combined = population + offspring
        combined.sort(key=lambda ind: tour_length(ind, dist_matrix))
        population = combined[:mu]

        best_per_gen.append(tour_length(population[0], dist_matrix))
        if gen % 10 == 0:
            print(f"Generation {gen}: Best = {best_per_gen[-1]:.2f}")
    return population[0], best_per_gen

# --- Tour & Konvergenz plotten ---
def plot_tour(tour, coords, history):
    tour_coords = coords[tour + [tour[0]]]
    plt.figure(figsize=(8, 5))
    plt.plot(tour_coords[:,0], tour_coords[:,1], 'o-')
    plt.title("Beste gefundene Tour")
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(history)
    plt.title("Konvergenzkurve")
    plt.xlabel("Generation")
    plt.ylabel("Tourlänge")
    plt.grid(True)
    plt.show()

def main():
    # --- Parameter ---
    tsp_file = "F:/DEV/PYHTONPROJECTS/TSP/data/eil51.tsp"
    mu = 200
    lam = 300
    mutation_rate = 0.05
    generations = 500

    # Einlesen & Distanzmatrix
    coords = read_tsplib(tsp_file)
    dist_mat = distance_matrix(coords)
    n = len(coords)

    # Startpopulation
    population = [create_individual(n) for _ in range(mu)]

    # Evolution
    best_ind, history = evolve(population, dist_mat, mu, lam, mutation_rate, generations)
    best_len = tour_length(best_ind, dist_mat)
    print(f"\nFinal tour length: {best_len:.2f}")

    # Plot
    plot_tour(best_ind, coords, history)

if __name__ == "__main__":
    main()
