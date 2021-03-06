from distanceUtils import DistanceUtils
from phase2.dimReductionEngineForVisual import DimReductionEngineForVisual


# Input files
class Task5:
    def __init__(self, base_path, database_operations):
        self._base_path = base_path;
        self._database_operations = database_operations
        self._dim_red_engineVis = DimReductionEngineForVisual(base_path, database_operations)

    def input(self):
        print("Enter  location ID and k")
        input_str = input()
        splitInput = input_str.split()
        location_id = splitInput[0]
        k = int(splitInput[1])
        print("Select method for reduction 1. PCA 2. SVD 3. LDA")
        input_method = int(input())
        (reduced_dimensions, post_projection_vectors, object_index_dict) = self._dim_red_engineVis.reduce_dimensions(
            input_method, 5, location_id, k)

        DistanceUtils.find_similar_entities_based_on_visual_descriptors(reduced_dimensions,
                                                                        post_projection_vectors,
                                                                        object_index_dict,
                                                                        int(location_id))
