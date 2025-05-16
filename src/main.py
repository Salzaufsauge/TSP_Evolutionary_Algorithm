import pathlib as path
import matplotlib.pyplot as plt
import import_data
from evolutionary_algorithm.evolutionary_algorithm import EvolutionaryAlgorithm


def get_project_root() -> path.Path:
    return path.Path(__file__).parent.parent

def main():
    tsp_files = get_project_root().glob('data/*.tsp')
    for tsp_file in tsp_files:
        data = import_data.read_tsp(tsp_file)
        algorithm = EvolutionaryAlgorithm(data[0], data[1],150,0.2,50,5,150)
        best_tour,history = algorithm.run(1500)
        if (len(best_tour[1]) - 1) == len(set(best_tour[1])):
            print(f"Best tour found for {tsp_file.name} is {best_tour[0]}")
        else:
            print(f"{len(best_tour[1])} == {len(set(best_tour[1]))}")
        tour_coords = [data[1][i] for i in best_tour[1]]
        plt.plot(*zip(*tour_coords),'b-',linewidth=2,label="Tour")
        plt.show()
        plt.plot(history)
        plt.title("Convergence Curve")
        plt.xlabel("Generation")
        plt.ylabel("Tour Length")
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    main()