import random
import numpy as np


class EvolutionaryAlgorithm:
    def __init__(self, weight_type, coords,initial_population_size,mutation_probability):
        self._coords = coords
        self._best_tour = None
        self.distance_method = weight_type
        self._distance_matrix = self._create_distance_matrix()
        self.population = self._create_initial_population(initial_population_size)
        self.mutation_probability = mutation_probability
        self._parents = None

    def run(self,iterations):
        for i in range(iterations):
            pass
        return self._best_tour

    def _create_initial_population(self,initial_population_size:int):
        population = []
        for i in range(initial_population_size):
            tour = [i for i in range(len(self._coords))]
            random.shuffle(tour)
            population.append(tour)
        return population

    def _variation(self):
        for i in range(random.randint(0,len(self._parents))):
            child = self._crossover(self._parents[i],self._parents[random.randint(0,len(self._parents)-1)])
            if random.random() < self.mutation_probability:
                self._mutate(child[0])
            if random.random() < self.mutation_probability:
                self._mutate(child[1])

    def _crossover(self, parent1, parent2):
        crossover_point = random.randint(1,len(self._coords)-2)
        parent_split_1 = np.array_split(parent1,[crossover_point])
        parent_split_2 = np.array_split(parent2,[crossover_point])
        return [np.concatenate((parent_split_1[0],parent_split_2[1])),np.concatenate((parent_split_2[0],parent_split_1[1]))]

    def _mutate(self, child):
        child

    def _calculate_distance(self,point1,point2):
        if point1 == point2:
            return 0
        match self.distance_method:
            case "EUC_2D" | "EUC_3D":
                return np.linalg.norm(point1-point2)
            case "MAN_2D" | "MAN_3D":
                return np.sum(np.abs(point1-point2))
            case "MAX_2D" | "MAX_3D":
                return np.max(np.abs(point1-point2))
            case _:
                raise ValueError("Weight type not implemented")

    def _create_distance_matrix(self):
        n = len(self._coords)
        dist_matrix = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                dist_matrix[i][j] = self._calculate_distance(self._coords[i],self._coords[j])
        return dist_matrix



