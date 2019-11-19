from pathlib import Path

import numpy as np
import pandas as pd
import scipy.spatial
import scipy.stats
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.sparse.csgraph._validation import validate_graph


class DistanceUtils:

    @staticmethod
    def cosine_similarity(target_vector, candidate_vector):
        # function that returns the cosine similarity of 2 vectors
        dot_product = np.dot(target_vector, candidate_vector)
        norm_a = np.linalg.norm(target_vector)
        norm_b = np.linalg.norm(candidate_vector)
        return dot_product / (norm_a * norm_b)

    # function to find similarity between entities on the basis of textual features using cosine similarity
    # Args:
    # @target_entity : vector of the query entity after transformation to low dimensional space
    # @candidate_entities: vectors of candidate objects in the DB after transformation to the lower dimensional space
    # @object_index_dict: corresponding vector space dictionary that maintains mapping of candidates to indexes
    @staticmethod
    def find_similarity_between_entities(target_entity, candidate_entities, candidate_index_dict, similarity_measure):
        l = 0

        result = pd.DataFrame(columns=['entity', 'similarity'])
        for i in range(len(candidate_entities)):
            if similarity_measure == 3:
                value = scipy.stats.entropy(target_entity, candidate_entities[i])
            else:
                value = DistanceUtils.cosine_similarity(target_entity, candidate_entities[i])
            result.loc[l] = [[key for key, entity in candidate_index_dict.items() if entity == i],
                             value]
            l = l + 1
        result = result.sort_values(by=['similarity'])
        if similarity_measure == 3:
            result = result.sort_values(['similarity'], ascending=True)
        else:
            result = result.sort_values(['similarity'], ascending=False)
        print(result.head())

    # function to find similarity between entities on the basis of visual descriptors using city-block
    # Args:
    # @projected_target_vector :  vector representing the query entity after transformation to low dimensional space
    # @projected_candidate_vectors: vectors of candidate objects in the DB after transformation to the
    # lower dimensional space
    # @object_index_dict: corresponding vector space dictionary that maintains mapping of candidates to indexes
    # @entity_id: id of the original entity
    @staticmethod
    def find_similar_entities_based_on_visual_descriptors(projected_target_vector, projected_candidate_vectors,
                                                          object_index_dict, entity_id):
        target_features = pd.DataFrame(projected_target_vector)
        similarity = pd.DataFrame(columns=['location', 'distance'])
        for i in range(1, 31):
            if i == entity_id:
                distances = scipy.spatial.distance.cdist(target_features.iloc[:, 0:],
                                                         target_features.iloc[:, 0:],
                                                         metric='cityblock')
                dist = np.sum(distances)/len(target_features)
            else:
                candidates_per_location = pd.DataFrame(projected_candidate_vectors.get(i))
                distances = scipy.spatial.distance.cdist(target_features.iloc[:, 0:],
                                                         candidates_per_location.iloc[:, 0:],
                                                         metric='cityblock')
                dist = np.sum(distances) / len(candidates_per_location)
            similarity.loc[i] = [object_index_dict.get(int(i)), dist]
        similarity = similarity.sort_values(['distance'], ascending=True)
        print(similarity.head())

    def find_similar_images_locations_for_given_model(image_id, k, model, reduced_space, count, loc_id_key_map):
        """ Gives 5 similar images and locations  for given model and image id
            Parameters
            ----------
            image_id : int
                       given by user
            k : int
                Number of Latent semantics(given by user)
            model : model given by user
            reduced_space : Reduced object and latent semantics matrix
            count: 5
                   Given in task

            Returns
            -------
            SimilarImage - 5 similar iamges for a given image id
            Similarloc - 5 similar locations for a given image id
        """
        count = int(count)
        primary_df = reduced_space
        search_loc_data_frame = primary_df[primary_df["image"] == image_id]
        search_loc_data_frame = search_loc_data_frame.iloc[:, 2:]
        search_loc_data_frame = search_loc_data_frame.head(1)
        distance_matrix = scipy.spatial.distance.cdist(primary_df.iloc[:, 2:],
                                                       search_loc_data_frame)  # taking only first row of the search image when multiple entries exist
        distance_space = pd.DataFrame(distance_matrix, columns=["dist"])
        SimilarityMatrix = reduced_space.merge(distance_space, left_index=True, right_index=True,
                                               how='inner')  # latent semantics merged with distance
        SimilarityMatrix = SimilarityMatrix.sort_values(['dist'], ascending=True)
        SimilarImage = SimilarityMatrix.drop_duplicates(['image']).head(count)
        # SimilarityMatrix.iloc[0:count,:]
        Similarloc = pd.DataFrame(columns=['locationId', 'dist'])
        for id in loc_id_key_map:
            arg_data_frame = SimilarityMatrix[SimilarityMatrix["locationId"] == id]
            avg_distance = arg_data_frame.loc[:, ['dist']].values.sum() / arg_data_frame['dist'].count()
            Similarloc.loc[id] = id, avg_distance
        Similarloc = Similarloc.sort_values(['dist'], ascending=True)
        return (SimilarImage, Similarloc.iloc[0:count, :])

    @staticmethod
    def get_dist_matrix(similarity_graph, base_path, image_count, similar_count):
        dist_matrix_file = Path(base_path + "/" + "dist_matrix" + str(image_count) + str(similar_count) + ".npy")
        if dist_matrix_file.is_file():
            dist_matrix = np.load(base_path + "/" + "dist_matrix" + str(image_count) + str(similar_count)+ ".npy")
        else:
            similarity_graph = np.maximum(similarity_graph, similarity_graph.T)
            similarity_graph = csr_matrix(similarity_graph)
            similarity_graph_sparse = similarity_graph
            similarity_graph_sparse = validate_graph(similarity_graph_sparse, False, np.float64)
            dist_matrix = shortest_path(similarity_graph_sparse, method="J", directed=False)
            np.save(base_path + "/" + "dist_matrix" + str(image_count) + str(similar_count) + ".npy", dist_matrix)
        return dist_matrix