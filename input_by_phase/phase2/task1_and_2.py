import numpy as np

from distanceUtils import DistanceUtils
from generalUtils import GeneralUtils
from phase2.dimReductionEngineForText import DimReductionEngine


# Input files
class Task1And2:
    def __init__(self, database_operations):
        self._database_operations = database_operations
        self._dim_red_engine = DimReductionEngine(database_operations)
        self._text_options = {1: "User Id", 2: "Image Id", 3: "Location Id"}
        self.model_options = {1: "TF_IDF", 2: "TF_IDF", 3: "TF"}

    def input(self):
        print("Select vector space  1.User Term 2. Image Term 3. Location Term")
        input_option = int(input())
        print("Enter number of latent semantics")
        input_str = input()
        splitInput = input_str.split()
        # model = 'TF_IDF'
        latent_semantics = splitInput[0]
        print("Select method for reduction 1. PCA 2. SVD 3. LDA")
        reduction_model = int(input())
        # reduce original dimensions to number of latent semantics as provided by the user.
        # function returns 3  arguments:
        # reduced_dimensions : selected vector space that has been projected onto the latent semantics identified.
        # projection_direction: vectors that denote latent semantics obtained after decomposition.
        # object_index_dict: dictionary that contains object index mapping
        # eg: term-index / image-index /location_id-location_name
        (reduced_dimensions, projection_direction, object_index_dict) = self._dim_red_engine.reduce_dimensions(
            reduction_model, input_option, self.model_options[reduction_model], latent_semantics)
        input_param = self._text_options[input_option]

        # TASK 2
        id_dict = {}
        print("Enter User Id, Image Id and Location Id")
        input_str = input();
        splitInput = input_str.split();
        id_dict[1] = splitInput[0]
        id_dict[2] = splitInput[1]
        id_dict[3] = splitInput[2]

        for option in self._text_options.keys():
            print(self._text_options.get(option).replace('Id', '') + ' Similarity')
            print('')
            if option == input_option:
                target_object = object_index_dict.get(id_dict.get(option))
                target_features = reduced_dimensions[target_object]
                DistanceUtils.find_similarity_between_entities(target_features, reduced_dimensions,
                                                               object_index_dict, reduction_model)

            if option != input_option:
                (vector_space, object_dict, term_index_dict) = \
                    self._dim_red_engine.get_vector_space(option, self.model_options[reduction_model])
                post_projection_vectors = GeneralUtils.get_projection(np.array(vector_space), projection_direction)
                target_object = object_dict.get(id_dict.get(option))
                target_features = post_projection_vectors[target_object]
                DistanceUtils.find_similarity_between_entities(target_features, post_projection_vectors,
                                                               object_dict, reduction_model)

            for option2 in self._text_options.keys():
                if option2 != option:
                    (vector_space, object_dict, term_index_dict) = \
                        self._dim_red_engine.get_vector_space(option2, self.model_options[reduction_model])
                    post_projection_vectors = GeneralUtils.get_projection(np.array(vector_space), projection_direction)
                    DistanceUtils.find_similarity_between_entities(target_features, post_projection_vectors,
                                                                   object_dict, reduction_model)
            print('')
