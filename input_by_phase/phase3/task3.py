from image_viewer.visualizer import Visualizer
from phase3.graphProcessor import GraphProcessor


class Task3:
    def __init__(self,base_path, database_ops):
        self.text_graph_processor = GraphProcessor(base_path, database_ops)
        self.visualizer = Visualizer(base_path, database_ops)
        self.display_choice = {1: self.visualize, 2: self.print}

    def visualize(self, image_index_list, obj_index):
        self.visualizer.visualize(image_index_list, obj_index)

    def print(self, image_index_list, obj_index):
        print(",".join(str(obj_index.iloc[i, 0]) for i in image_index_list))

    def input(self, obj_index, sim_graph, k):
        result = self.text_graph_processor.get_page_rank_without_transition_matrix(sim_graph, k)
        print("Input K, number of dominant nodes needed, for graph")
        k = int(input())

        continue_choice = "Y"
        while continue_choice != "N":
            print("Select 1. Visualize 2. Print")
            input_choice = int(input())
            self.display_choice[input_choice](result[:k], obj_index)
            print("Press N to stop displaying results")
            continue_choice = input()
