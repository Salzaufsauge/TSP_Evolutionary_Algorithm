
import pathlib as path
import import_data
from src.evolutionary_algorithm.distance_matrix import distance_matrix

from src.import_data import read_tsp


def get_project_root() -> path.Path:
    return path.Path(__file__).parent.parent

def main():
    tsp_files = get_project_root().glob('data/*.tsp')
    for tsp_file in tsp_files:
        print(import_data.read_tsp(tsp_file))


    coords = read_tsp("eil51.tsp")
    dist_matrix = distance_matrix(coords)
    n = len(coords)

    mu = 50
    lam = 100
    mutation_rate = 0.1
    generations = 100


if __name__ == '__main__':
    main()