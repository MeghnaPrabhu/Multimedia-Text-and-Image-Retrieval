from distanceUtils import DistanceUtils
from input_by_phase.phase2.task1_and_2 import Task1And2
from input_by_phase.phase2.task5 import Task5
from input_by_phase.phase2.task6 import Task6
from input_by_phase.phase2.task7 import Task7
from input_by_phase.phase2.task3 import Task3
from input_by_phase.phase2.task4 import Task4
from phase1.inputParameter import InputParameter;
from phase2.dimReductionEngineForText import DimReductionEngine


class Phase2:
    def __init__(self, base_path, database_operations):
        self.base_path = base_path
        self._database_operations = database_operations
        self._dim_red_engine = DimReductionEngine(database_operations)
        # TODO: Needs to be refactored

        self.tasks = {1: Task1And2(database_operations),
                      3: Task3(base_path, database_operations),
                      4: Task4(base_path, database_operations),
                      5: Task5(base_path, database_operations),
                      6: Task6(base_path, database_operations),
                      7: Task7(database_operations)}

    def input(self):
        print("Select Tasks  1.Task 1 & 2 3. Task 3 4. Task 4 5.Task 5 6. Task 6 7. Task 7")
        task = int(input())
        self.tasks[task].input()
