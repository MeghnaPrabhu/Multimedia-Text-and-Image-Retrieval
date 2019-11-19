from phase2.dimReductionEngineForText import DimReductionEngine

class Task6:
    def __init__(self, base_path, db_operations):
        self._base_path = base_path;
        self._db_operations = db_operations
        self.dim_red_engine = DimReductionEngine(db_operations)

    def input(self):
        print("Enter value of k (number of latent semantics for SVD)")
        k = int(input())
        self.dim_red_engine.get_location_location_similarity_matrix_and_reduce(k)

