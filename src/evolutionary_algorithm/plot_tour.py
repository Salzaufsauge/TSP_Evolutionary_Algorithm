import matplotlib.pyplot as plt

def plot_tour(tour, coords):
    path = coords[tour + [tour[0]]]
    plt.plot(path[:,0], path[:,1], 'o-')
    plt.title("Best Tour")
    plt.show()