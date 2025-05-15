def tour_lenght(tour, dist_matrix):

    return sum(dist_matrix[tour[i], tour[(i+1)%len(tour)]] for i in range(len(tour)))
