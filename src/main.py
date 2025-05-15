
import pathlib as path

from matplotlib import pyplot as plt

import import_data
from src.evolutionary_algorithm.distance_matrix import distance_matrix
from src.evolutionary_algorithm.evolve import evolve
from src.evolutionary_algorithm.generate_random_paths import generate_random_paths
from src.evolutionary_algorithm.plot_tour import plot_tour
from src.evolutionary_algorithm.tour_lenght import tour_lenght

from src.import_data import read_tsp


def get_project_root() -> path.Path:
    return path.Path(__file__).parent.parent

def main():
    tsp_files = get_project_root().glob('data/*.tsp')
    for tsp_file in tsp_files:
        print(import_data.read_tsp(tsp_file))


    coords = read_tsp("bier127.tsp")
    dist_matrix = distance_matrix(coords)
    n = len(coords)

    mu = 50
    lam = 100
    mutation_rate = 0.1
    generations = 100

# Starte mit zufälliger Population
    population = [generate_random_paths(n) for _ in range(mu)]

    # Führe den EA durch
    best_ind, history = evolve(population, dist_matrix, mu, lam, mutation_rate, generations)

    print("Final tour length:", tour_lenght(best_ind, dist_matrix))  # Ausgabe beste Tourlänge

    plot_tour(best_ind, coords)  # Visualisiere beste Tour

    # Zeige Konvergenzkurve (Verlauf der besten Fitnesswerte)
    plt.plot(history)
    plt.title("Convergence Curve")
    plt.xlabel("Generation")
    plt.ylabel("Tour Length")
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()