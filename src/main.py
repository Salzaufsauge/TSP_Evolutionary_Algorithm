import pathlib as path
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import import_data
from evolutionary_algorithm.evolutionary_algorithm import EvolutionaryAlgorithm


def get_project_root() -> path.Path:
    return path.Path(__file__).parent.parent

def run_algorithm(tsp_data):
    t0 = time.time()
    algorithm = EvolutionaryAlgorithm(tsp_data[0], tsp_data[1], 1500, 0.1, 1000, 5, 400)
    best_tour, history = algorithm.run(3000)
    t1 = time.time()
    print(f"finished run best tour is {best_tour[0]} in {t1 - t0:.2f}s")
    return history

def main():
    # tsp_files = get_project_root().glob('data/*.tsp')
    tsp_file = get_project_root().joinpath('data/bier127.tsp')
    tsp_data = import_data.read_tsp(tsp_file)

    t2 = time.time()

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_algorithm, tsp_data) for _ in range(10)]
        histories = [f.result() for f in futures]

    average_history = [sum(histories) / len(histories) for histories in zip(*histories)]
    plt.figure(figsize=(10, 6))

    for i, history in enumerate(histories):
        plt.plot(history, color='lightgray', linewidth=1, label=f'Run {i}' )

    plt.plot(average_history, color='red', linewidth=2, label='Average')

    N = 200
    for i in range(0, len(average_history), N):
        plt.annotate(f"{average_history[i]:.0f}", (i, average_history[i]), textcoords="offset points", xytext=(0, 10),
                     ha='center', fontsize=8)

    plt.title("Convergence Curve")
    plt.xlabel("Generation")
    plt.ylabel("Tour Length")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    t3 = time.time()
    print(f"Total time: {t3 - t2:.2f}s")
    # for tsp_file in tsp_files:
    #     t0 = time.time()
    #     data = import_data.read_tsp(tsp_file)
    #     algorithm = EvolutionaryAlgorithm(data[0], data[1], 1000, 0.1, 1000, 5, 400)
    #     best_tour, history = algorithm.run(1500)
    #     print(f"Best tour found for {tsp_file.name} is {best_tour[0]}")
    #     # tour_coords = [data[1][i] for i in best_tour[1]]
    #     # plt.plot(*zip(*tour_coords),'b-',linewidth=2,label="Tour")
    #     # plt.show()
    #     plt.plot(history)
    #     plt.title("Convergence Curve")
    #     plt.xlabel("Generation")
    #     plt.ylabel("Tour Length")
    #     plt.grid(True)
    #     plt.show()
    #
    #     t1 = time.time()
    #     print(t1 - t0)

if __name__ == '__main__':
    main()