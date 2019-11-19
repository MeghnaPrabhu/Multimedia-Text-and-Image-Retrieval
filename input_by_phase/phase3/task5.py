from pprint import pprint

import numpy as np

from image_viewer.visualizer import Visualizer
from phase3.LSH import lshOrchestrator


class Task5a:
    def __init__(self, base_path, database_ops):
        self.lshOrchestrator = lshOrchestrator(base_path, database_ops)

    def input(self):
        # inp_vect = input('Enter input vector\n')
        input_vec = np.random.randint(low=0, high=256, size=(19, 9))
        idx = np.arange(20).reshape(20, 1)
        input_vec = np.vstack((input_vec, input_vec[0]))
        input_vec = np.hstack((idx, input_vec))
        num_layer_hash = input('Enter number of layers, hash : ')
        num_layer, num_hash = map(int, num_layer_hash.split())
        hashTable = self.lshOrchestrator.run_lsh(input_vec, num_layers=num_layer, num_hash=num_hash)
        for layer_idx, layer in enumerate(hashTable):
            print(f'Layer {layer_idx + 1}')
            print('=' * 30)
            pprint(layer)


class Task5b:
    def __init__(self, base_path, database_ops):
        self.lshOrchestrator = lshOrchestrator(base_path, database_ops)
        self.visualizer = Visualizer(base_path, database_ops)

    def input(self):
        image_t = input('Enter ImageId & t : ')
        query, t = image_t.split()
        candidates = self.lshOrchestrator.img_ann(query=query, k=int(t))
        pprint(candidates)
        self.visualizer.visualize_with_ids(candidates)
