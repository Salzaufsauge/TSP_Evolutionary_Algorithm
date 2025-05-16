import math
import random
import numpy as np


def nint(val):
    return int(val + 0.5)


def calc_lat_and_longitude(point):
    deg_lat = nint(point[0])
    min_lat = point[0] - deg_lat
    lat = math.pi * (deg_lat + 5.0 * min_lat / 3.0) / 180.0

    deg_long = nint(point[1])
    min_long = point[1] - deg_long
    long = math.pi * (deg_long + 5.0 * min_long / 3.0) / 180.0
    return lat, long


def calc_geo_distance(point1, point2):
    rrr = 6378.388
    longitude1, latitude1 = calc_lat_and_longitude(point1)
    longitude2, latitude2 = calc_lat_and_longitude(point2)
    q1 = math.cos(longitude1 - longitude2)
    q2 = math.cos(latitude1 - latitude2)
    q3 = math.cos(latitude1 + latitude2)
    return rrr * math.acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0


class EvolutionaryAlgorithm:
    def __init__(self, weight_type, coords,initial_population_size,mutation_probability,mu,step_length,lam):
        self._coords = coords
        self._best_tour = None
        self.distance_method = weight_type
        self._distance_matrix = self._create_distance_matrix()
        self.population = self._create_initial_population(initial_population_size)
        self._mutation_probability = mutation_probability
        self._lam = lam
        self._mu = mu
        self._sigma = step_length
        self._parents = None
        self._children = None
        self._history = []

    def run(self,iterations):
        tour_lengths = self._evaluate_population()
        for i in range(iterations):
            self._parents = self._selection(tour_lengths)
            self._variation()
            tour_lengths = self._evaluate_population()
            self.population = self._selection(tour_lengths) + self._children
            self._evaluate_population()
            self._history.append(self._best_tour[0])
        return self._best_tour, self._history

    def _create_initial_population(self,initial_population_size:int):
        population = []
        for _ in range(initial_population_size):
            tour = [i for i in range(len(self._coords))]
            random.shuffle(tour)
            population.append(tour)
        return population

    def _selection(self,tour_lengths):
        list1, list2 = zip(*sorted(zip(tour_lengths, self.population)))
        list2 = list(list2)
        return list2[:self._mu]

    def _evaluate_population(self):
        tour_lengths = []
        best_distance = None
        for tour in self.population:
            tour_distance = 0.0
            roundtrip = tour + [tour[0]]
            for i in range(1,len(roundtrip)):
                tour_distance += float(self._distance_matrix[roundtrip[i-1]][roundtrip[i]])
            tour_lengths.append(tour_distance)
            if best_distance is None or tour_distance < best_distance:
                best_distance = tour_distance
                self._best_tour = (tour_distance,roundtrip.copy())
        return tour_lengths

    def _variation(self):
        children = []
        for _ in range(self._lam):
            parents = random.sample(self._parents, 2)
            child = self._crossover(parents[0], parents[1])
            if random.random() < self._mutation_probability:
                self._mutate(child[0])
            if random.random() < self._mutation_probability:
                self._mutate(child[1])
            children.extend(child)
        self._children = children

    def _crossover(self, parent1, parent2):
        crossover_point1 = random.randint(1,len(self._coords)//3*2)
        crossover_point2 = random.randint(crossover_point1 + 1,len(self._coords)-2)

        split_points = [0] + [crossover_point1,crossover_point2] + [len(self._coords)]

        parent_split_1 = [parent1[split_points[i]:split_points[i+1]] for i in range(len(split_points)-1)]
        parent_split_2 = [parent2[split_points[i]:split_points[i+1]] for i in range(len(split_points)-1)]

        empty_child1 = [None] * len(parent_split_1[0]) + parent_split_1[1] + [None] * len(parent_split_1[2])
        empty_child2 = [None] * len(parent_split_2[0]) + parent_split_2[1] + [None] * len(parent_split_2[2])

        known1 = {val for val in empty_child1 if val is not None}
        known2 = {val for val in empty_child2 if val is not None}

        values1 = (val for val in parent2 if val not in known1)
        values2 = (val for val in parent1 if val not in known2)

        child1 = [next(values1) if val is None else val for val in empty_child1]
        child2 = [next(values2) if val is None else val for val in empty_child2]

        return [child1,child2]

    def _mutate(self, child):
        for i in range(len(child) - self._sigma):
            if random.randint(0,1):
                child[i],child[i+self._sigma] = child[i+self._sigma],child[i]

    def _calculate_distance(self,point1,point2) -> int:
        match self.distance_method:
            case "EUC_2D" | "EUC_3D":
                return nint(np.linalg.norm(point1 - point2))
            case "MAN_2D" | "MAN_3D":
                return nint(np.sum(np.abs(point1 - point2)))
            case "MAX_2D" | "MAX_3D":
                return max(nint(point1), nint(point2))
            case "GEO":
                return int(calc_geo_distance(point1, point2))
            case _:
                raise ValueError("Weight type not implemented")

    def _create_distance_matrix(self):
        n = len(self._coords)
        dist_matrix = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                dist_matrix[i][j] = self._calculate_distance(self._coords[i],self._coords[j])
        return dist_matrix
