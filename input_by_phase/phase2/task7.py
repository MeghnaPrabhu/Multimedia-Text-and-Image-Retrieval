from phase2.dimReductionEngineForText import DimReductionEngine
from phase2.tensorProcessor import TensorProcessor

class Task7:

    def __init__(self, database_operations):
        self._database_operations = database_operations
        self._dim_red_engine = DimReductionEngine(database_operations)
        self.tensor_processor = TensorProcessor(database_operations)

    def input(self):
        print("Enter value of k (number of latent semantics for decomposition)")
        k = int(input())
        tensor = self._dim_red_engine.get_tensor()
        self.tensor_processor.process_tensor(tensor, k)
