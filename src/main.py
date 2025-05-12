import numpy as np
import pathlib as path
import import_data

def get_project_root() -> path.Path:
    return path.Path(__file__).parent.parent

def main():
    tsp_files = get_project_root().glob('data/*.tsp')
    for tsp_file in tsp_files:
        print(import_data.read_tsp(tsp_file))

if __name__ == '__main__':
    main()