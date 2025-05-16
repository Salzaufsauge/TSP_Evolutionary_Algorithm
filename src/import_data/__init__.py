import numpy as np

# reads tsp files for now only supports 2D tsps
def read_tsp(file):
    with open(file, 'r') as f:
        lines = f.readlines()
        weight_type = None
        found_node_coord_section = False
        coords = []
        for line in lines:
            if line.startswith('EDGE_WEIGHT_TYPE'):
                weight_type = line.split(':')[1].strip()
                continue
            if line.startswith('NODE_COORD_SECTION'):
                found_node_coord_section = True
                continue
            if found_node_coord_section:
                if line.startswith('EOF'):
                    break
                node_id, x, y = line.split()
                coords.append((float(x), float(y)))
        return weight_type, np.array(coords)