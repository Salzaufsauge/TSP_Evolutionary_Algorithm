import numpy as np


def distance_matrix(coords):

    n = len(coords)                 # Matrix mit Länge ( Anzahl der Städte ) erstellen
    distance = np.zeros((n, n))     # Leere Matrix erzeugen
    for i in range(n):
        for j in range(n):
            distance[i, j] = np.linalg.norm(coords[i] - coords[j]) # euklidische Distanz

    return distance