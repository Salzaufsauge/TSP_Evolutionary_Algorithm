import numpy as np
import pathlib as path
import import_data
from evolutionary_algorithm import EvolutionaryAlgorithm


def get_project_root() -> path.Path:
    return path.Path(__file__).parent.parent

def main():
    tsp_files = get_project_root().glob('data/*.tsp')
    for tsp_file in tsp_files:
        data = import_data.read_tsp(tsp_file)
        algoritm = EvolutionaryAlgorithm(data[0], data[1],0.5)
        best_tour = algoritm.run()

if __name__ == '__main__':
    main()