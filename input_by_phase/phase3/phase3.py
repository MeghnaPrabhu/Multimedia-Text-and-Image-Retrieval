from input_by_phase.phase3.task1To4 import Task1To4
from input_by_phase.phase3.task5 import Task5a
from input_by_phase.phase3.task5 import Task5b
from input_by_phase.phase3.task6 import Task6


class Phase3:
    def __init__(self, base_path, database_operations):
        self.base_path = base_path
        self._database_operations = database_operations
        self.tasks = {'1': Task1To4(base_path, database_operations), '5a': Task5a(base_path, database_operations),
                      '5b': Task5b(base_path, database_operations), '6': Task6(base_path, database_operations)}

    def input(self):
        print("Select Tasks  1.Task 1, 2, 3, 4 \t 5a.Task 5.a \t 5b.Task 5.b \t 6. Task 6")
        task = input()
        self.tasks[task].input()

