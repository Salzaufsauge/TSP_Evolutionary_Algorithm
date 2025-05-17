import numpy as np  # Importiert NumPy, wichtig für numerische Berechnungen (z. B. Abstände)
import random       # Für zufällige Operationen wie Shuffle, Mutationen etc.
import matplotlib.pyplot as plt  # Zum Visualisieren der Ergebnisse (Tour, Fitnessverlauf)

# --- Funktion zum Einlesen einer TSP-Datei im TSPLIB-Format ---
def read_tsplib(filename):
    with open(filename, 'r') as f:  # Öffnet die Datei im Lesemodus
        coords = []  # Liste für die Koordinaten der Städte
        found = False  # Schalter: Haben wir die "NODE_COORD_SECTION" gefunden?
        for line in f:  # Durchläuft jede Zeile der Datei
            if 'NODE_COORD_SECTION' in line:  # Startsignal für Koordinaten
                found = True
                continue  # Zur nächsten Zeile
            if found:
                if 'EOF' in line:  # Ende der Datei erreicht
                    break
                parts = line.strip().split()  # Zerlegt die Zeile in Einzelteile
                if len(parts) >= 3:  # Erwartet mindestens Index, x, y
                    coords.append((float(parts[1]), float(parts[2])))  # x, y speichern
        return np.array(coords)  # Gibt Koordinaten als NumPy-Array zurück

# --- Funktion zur Berechnung der Distanzmatrix ---
def distance_matrix(coords):
    n = len(coords)  # Anzahl Städte
    dist = np.zeros((n, n))  # Leere Matrix n×n
    for i in range(n):
        for j in range(n):
            dist[i][j] = np.linalg.norm(coords[i] - coords[j])  # Euklidische Distanz
    return dist  # Gibt die fertige Distanzmatrix zurück

# --- Funktion zur Berechnung der Tourlänge ---
def tour_length(tour, dist_matrix):
    # Berechnet die Gesamtlänge einer Rundreise durch alle Städte in tour[]
    return sum(dist_matrix[tour[i], tour[(i+1)%len(tour)]] for i in range(len(tour)))

# --- Erstellt ein zufälliges Individuum (d. h. eine zufällige Tour) ---
def create_individual(n):
    ind = list(range(n))  # Erzeugt eine Liste [0, 1, ..., n-1]
    random.shuffle(ind)  # Mischt die Liste zufällig (Permutation)
    return ind  # Gibt die zufällige Tour zurück

# --- Order Crossover (OX) für zwei Eltern-Individuen ---
def order_crossover(p1, p2):
    size = len(p1)  # Länge der Tour (Anzahl Städte)
    a, b = sorted(random.sample(range(size), 2))  # Zufällige Crossover-Grenzen
    hole = set(p1[a:b+1])  # Teilstück aus Elternteil 1 merken
    child = [None]*size  # Leeres Kind initialisieren
    child[a:b+1] = p1[a:b+1]  # Teilstück übernehmen

    pos = (b+1)%size  # Startposition zum Einfügen
    for city in p2:  # Durchlaufe Elternteil 2
        if city not in hole:  # Nur Städte, die noch nicht im Kind sind
            while child[pos] is not None:  # Finde nächste freie Position
                pos = (pos + 1) % size
            child[pos] = city  # Stadt einfügen
    return child  # Gibt das neue Kind zurück

# --- Swap-Mutation (zwei Städte tauschen) ---
def swap_mutation(ind, mutation_rate):
    if random.random() < mutation_rate:  # Mutieren mit gegebener Wahrscheinlichkeit
        a, b = random.sample(range(len(ind)), 2)  # Zwei zufällige Positionen wählen
        ind[a], ind[b] = ind[b], ind[a]  # Städte tauschen
    return ind  # Gibt mutiertes Individuum zurück

# --- Haupt-EA: führt Evolution durch ---
def evolve(population, dist_matrix, mu, lam, mutation_rate, generations):
    best_per_gen = []  # Speichert besten Fitnesswert pro Generation
    for gen in range(generations):  # Schleife über alle Generationen
        offspring = []  # Liste für Nachkommen
        for _ in range(lam):  # Erzeuge λ Nachkommen
            p1, p2 = random.sample(population, 2)  # Wähle zwei Eltern zufällig
            child = order_crossover(p1, p2)  # Erzeuge Kind mit Crossover
            child = swap_mutation(child, mutation_rate)  # Mutiere das Kind
            offspring.append(child)  # Füge Kind zur Nachkommenliste hinzu

        combined = population + offspring  # Eltern und Kinder zusammenlegen
        combined.sort(key=lambda ind: tour_length(ind, dist_matrix))  # Nach Fitness sortieren
        population = combined[:mu]  # Beste μ Individuen überleben (μ + λ Selektion)

        best_per_gen.append(tour_length(population[0], dist_matrix))  # Besten Wert speichern

        if gen % 10 == 0:  # Optional: alle 10 Generationen Fortschritt ausgeben
            print(f"Generation {gen}: {best_per_gen[-1]:.2f}")
    return population[0], best_per_gen  # Rückgabe: bestes Individuum + Fitnessverlauf

# --- Zeigt die beste Tour grafisch an ---
def plot_tour(tour, coords):
    path = coords[tour + [tour[0]]]  # Schließt Rundtour (zurück zum Start)
    plt.plot(path[:,0], path[:,1], 'o-')  # Zeichnet Linien zwischen Städten
    plt.title("Best Tour") # Titel für das Diagramm
    plt.show()  # Zeigt das Diagramm an

# --- Hauptfunktion ---
def main():
    coords = read_tsplib("F:/DEV/PYHTONPROJECTS/TSP/data/bier127.tsp")  # Lies Koordinaten aus Datei
    dist_matrix = distance_matrix(coords)  # Berechne Distanzmatrix
    n = len(coords)  # Anzahl Städte

    # Parameter des EA (können später variiert werden)
    mu = 50             # Anzahl Eltern (Selektion)
    lam = 100           # Anzahl Nachkommen
    mutation_rate = 0.1  # Wahrscheinlichkeit für Mutation
    generations = 10000    # Wie viele Generationen laufen

    # Starte mit zufälliger Population
    population = [create_individual(n) for _ in range(mu)]

    # Führe den EA durch
    best_ind, history = evolve(population, dist_matrix, mu, lam, mutation_rate, generations)

    print("Final tour length:", tour_length(best_ind, dist_matrix))  # Ausgabe beste Tourlänge

    plot_tour(best_ind, coords)  # Visualisiere beste Tour

    # Zeige Konvergenzkurve (Verlauf der besten Fitnesswerte)
    plt.plot(history)
    plt.title("Convergence Curve")
    plt.xlabel("Generation")
    plt.ylabel("Tour Length")
    plt.grid(True)
    plt.show()

# --- Startpunkt des Programms ---
if __name__ == "__main__":
    main()  # Führt das Hauptprogramm aus
