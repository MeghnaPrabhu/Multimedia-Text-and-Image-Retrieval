from input_by_phase.phase3.task1 import Task1
from input_by_phase.phase3.task2 import Task2
from input_by_phase.phase3.task3 import Task3
from input_by_phase.phase3.task4 import Task4
from phase3.graphProcessor import GraphProcessor


class Task1To4:
    def __init__(self, base_path, database_ops):
        self.text_graph_processor = GraphProcessor(base_path, database_ops)
        self.tasks = {1: Task1(), 2: Task2(base_path, database_ops), 3: Task3(base_path, database_ops), 4: Task4(base_path, database_ops)}

    def input(self):
        print("Input number of outgoing edges k:")
        k = int(input())
        # unweighted graph func also available
        obj_index, sim_graph = self.text_graph_processor.get_weighted_image_similarity_graph(k)

        keep_running = True
        while keep_running:
            print("Select Tasks  1.)Task 1   2.)Task 2   3.) Task 3   4.) Task 4")
            input_option = int(input())
            self.tasks[input_option].input(obj_index, sim_graph, k)
            print("Press N to stop")
            input_option = input()
            if input_option == "N":
                keep_running = False



