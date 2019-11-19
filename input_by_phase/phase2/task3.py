from phase2.dimReductionEngineForVisual import DimReductionEngineForVisual


# Input files
class Task3:
    def __init__(self, base_path, database_operations):
        self._base_path = base_path;
        self._database_operations = database_operations
        self._dim_red_engineVis = DimReductionEngineForVisual(base_path, database_operations)

    def input(self):
        print("Enter  Model and k")
        input_str = input()
        splitInput = input_str.split()
        model = splitInput[0]
        k = int(splitInput[1])
        print("Select algorithm for 1.PCA 2.SVD 3.LDA")
        input_option = int(input())
        self._dim_red_engineVis.reduce_dimensions_givenmodel(input_option, model, k, 5)
