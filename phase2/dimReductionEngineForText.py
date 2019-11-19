import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
from scipy.spatial import distance
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation, PCA

from dbUtils import DBUtils
from generalUtils import GeneralUtils


class DimReductionEngine:
    def __init__(self, database_operations):
        self._database_operations = database_operations
        self.base_path = sys.argv[1]
        self.tensor_file_path = self.base_path + "/tensor.npy"
        self.tables_by_input = {1: ["termsPerUser", "userId"], 2: ["termsPerImage", "imageId"],
                                3: ["termsPerLocation", "locationId"]}
        self.tensor_factor_modes = {"user": ["userId", "termsPerUser"], "image": ["imageId", "termsPerImage"],
                                    "location": ["locationId", "termsPerLocation"]}
        self.reduction_method = {1: self.PCA, 2: self.SVD, 3: self.LDA}

    def create_index_tensor_factor(self, tensor_factor_mode):
        """ Create index for tensor factor matrix given a query
                                                            Parameters
                                                            ----------
                                                            tensor_factor_mode : The factor matrix type
                                                            Returns
                                                            -------
                                                            Indexed list of items
        """
        factor_mode_values = self.tensor_factor_modes[tensor_factor_mode]
        entity_id = factor_mode_values[0]
        table = factor_mode_values[1]
        query = "select distinct {0} from {1}".format(entity_id, table)
        entity_list = []
        result = self._database_operations.executeSelectQuery(query)
        for item in result:
            entity_list.append(item[0])

        return entity_list

    def create_index(self, query):

        """ Create index for tensor creation
                                                    Parameters
                                                    ----------
                                                    query : Query to create index
                                                    Returns
                                                    -------
                                                    Ordered dictionary of items from db
        """
        dict = OrderedDict()
        result = self._database_operations.executeSelectQuery(query)
        index = 0
        for item in result:
            dict[item[0]] = index
            index += 1

        return dict

    def get_tensor(self):
        """ Creates tensor for user, images and locations if already not present in file

            Parameters
            ----------
            Returns
            -------
            A multi dimensional array which represents the tensor: int
        """
        user_index = self.create_index("select distinct userId from termsPerUser");
        image_index = self.create_index("select distinct imageId from termsPerImage");
        loc_index = self.create_index("select distinct locationId from termsPerLocation");
        tensor_file = Path(self.tensor_file_path)
        tensor = np.zeros(shape=(len(user_index), len(image_index), len(loc_index)))
        if not tensor_file.is_file():
            get_shared_terms_query = "select termsPerUser.userId, termsPerImage.imageId, termsPerLocation.locationId, count(distinct termsPerUser.term) from termsPerUser " \
                                     "INNER JOIN termsPerImage on termsPerUser.term = termsPerImage.term INNER JOIN termsPerLocation on termsPerImage.term=termsPerLocation.term " \
                                     "group by termsPerUser.userId,termsPerImage.imageId,termsPerLocation.locationId"
            self._database_operations.executeSelectQueryWithRowProcessing(get_shared_terms_query,
                                                                          self.create_tensor_from_db_data_row_processor,
                                                                          [user_index, image_index, loc_index, tensor],
                                                                          True)
            np.save(self.tensor_file_path, tensor)
        else:
            tensor = np.load(self.tensor_file_path)

        return tensor

    def get_location_location_similarity_matrix_and_reduce(self, k):
        """ Creates a location location similarity matrix based on cosine and reduces it
            Parameters
            ----------
            k : int
            number of dimensions to be reduced to
            Returns
            -------
            void
        """
        (vector_space, object_index_dict, term_index_dict) = self.get_vector_space(3, "TF_IDF", True);
        np_vector_space = np.array(vector_space)
        distance_matrix = distance.cdist(np_vector_space, np_vector_space, metric='cosine')
        subtract_func = lambda t: 1 - t
        vfunc = np.vectorize(subtract_func)
        distance_matrix = vfunc(distance_matrix)
        (reduced_dimensions, projection_direction) = self.reduction_method[2](distance_matrix, k)
        location_index = DBUtils.create_location_id_key_map(self._database_operations)
        self.display_topics(projection_direction, location_index)

    def display_topics(self, components, location_map):
        """ Display function for printing latent semantics for location-location matrix
            Parameters
            ----------
            components : Array
            Reduced latent semantics
            location_map : map of location id to location name
            map containing location ids and keys
            Returns
            -------
            void
        """
        for topic_idx, topic in enumerate(components):
            print("\nTopic {0}:".format(topic_idx))
            res = [location_map[i + 1] + ":" + str(topic[i]) + "\n" for i in topic.argsort()[::-1]]
            print(" ".join(res[i] for i in range(len(location_map))))

    def create_tensor_from_db_data_row_processor(self, params, item):
        """ Adds tensor elements from sql results
            Parameters
            ----------
            params : Array
            params include items to help in processing tensor
            item : Array
                database row to be processed

            Returns
            -------
            None
        """
        user_index = params[0]
        image_index = params[1]
        loc_index = params[2]
        tensor = params[3]
        user_index_val = user_index[item[0]]
        image_index_val = image_index[item[1]]
        loc_index_val = loc_index[int(item[2])]
        tensor[user_index_val][image_index_val][loc_index_val] = int(item[3])

    def get_total_term_count(self, table_name, prune_stop_words=True):

        """ Gets total term count

            Parameters
            ----------
            table_name : str
                table to use (termsPerUser, termsPerImage, termsPerLocation)
            prune_stop_words : bool
                 optional parameter to control pruning of stop words

            Returns
            -------
            total terms in system : int

        """
        query_addage_for_term_pruning = " where length(term) > 2" if prune_stop_words else ""
        query_addage_for_rare_term_pruning = " AND term NOT IN (select term from termsPerImage  group by term having count(distinct imageId) <=5)" if prune_stop_words else ""
        get_terms_query = "select distinct term from terms{1}{2}".format(table_name, query_addage_for_term_pruning,
                                                                       query_addage_for_rare_term_pruning);
        result = self._database_operations.executeSelectQuery(get_terms_query)
        term_list = {}
        index = 0
        for item in result:
            term_list[item[0]] = index
            index += 1
        return term_list

    def get_vector_space(self, input, model, prune_stop_words=True, get_object_index=False):
        """ Generic function that creates vector-space given a model and input parameter

            Parameters
            ----------
            input : int
                 input to determine table to use (1 : user-term, 2: image-term, 3: location-term)
            model : str
                 text model to use (TF, DF, TF_IDF)
            prune_stop_words : bool
                  optional parameter to control pruning of stop words

            Returns
            -------
            vector space containing data : Array

        """
        table_name = self.tables_by_input[input][0]
        primary_field = self.tables_by_input[input][1]
        term_index_dict = self.get_total_term_count(table_name, prune_stop_words)
        total_term_count = len(term_index_dict)
        #eliminating terms with length lesser than 2 and words that were only used in lesser than 6 images
        query_addage_for_rare_term_pruning = " AND term NOT IN (select term from termsPerImage  group by term having count(distinct imageId) <=5)" if prune_stop_words else ""
        query_addage_for_term_pruning = " where length(term) > 2" if prune_stop_words else ""

        get_grouped_terms_query = "select distinct {0}, GROUP_CONCAT(newVal, \'|\') AS terms from (select {0}, term || \' \' || {1} as newVal from {2}{3}{4}) GROUP BY {0};".format(
            primary_field, model, table_name, query_addage_for_term_pruning, query_addage_for_rare_term_pruning)

        query_result = self._database_operations.executeSelectQuery(get_grouped_terms_query)

        object_index_dict = {}
        object_index_array = []
        object_index_value = 0;
        term_index_value = 0;
        total_obj = len(query_result)
        vector_space = [[0 for j in range(0, total_term_count)] for i in range(0, total_obj)]
        conversion_func = GeneralUtils.get_conversion_func_for_text_model(model)
        term_list_indexed = list(term_index_dict.keys())
        for item in query_result:
            object = str(item[0])
            concatenated_str = item[1]
            object_index_dict[object] = object_index_value;
            object_index_array.append(object)

            term_list = concatenated_str.split("|")
            for term_model_str in term_list:
                term_model_split_str = term_model_str.split()
                term = term_model_split_str[0]
                model_value = conversion_func(term_model_split_str[1])
                if term not in term_index_dict:
                    term_index_dict[term] = term_index_value
                    term_index = term_index_value
                    term_list_indexed.append(term)
                    term_index_value += 1
                else:

                    term_index = term_index_dict[term]
                vector_space[object_index_value][term_index] = model_value
            object_index_value += 1

        if not get_object_index:
            return vector_space, object_index_dict, term_list_indexed
        else:
            return vector_space, object_index_dict, object_index_array

    def LDA(self, vector_space, k):
        """ Performs LDA on vector space and reducing the dimensions to k from the original data.

            Parameters
            ----------
            vector_space : list
                vector space created on TF model using either user / image / location space as per user input
            k : int
                Number of Latent semantics(given by user)
            Returns
            -------
            lda.fit_transform(vector_space) - vector space after transformation
            lda.components_ - latent semantic in terms of original features

        """
        vector_space_np = np.array(vector_space)
        lda = LatentDirichletAllocation(n_components=int(k))
        res = lda.fit_transform(vector_space_np)
        return res, lda.components_

    def SVD(self, vector_space, k):
        """ Performs SVD on vector space and reducing the dimensions to k from the original data.

            Parameters
            ----------
            vector_space : list
                vector space created on TF_IDF model using either user / image / location space as per user input
            k : int
                Number of Latent semantics(given by user)

            Returns
            -------
            svd.fit_transform(vector_space) - vector space after transformation
            svd.components_ - latent semantic x features matrix (VT matrix)

        """
        svd = TruncatedSVD(n_components=int(k))
        svd.fit(vector_space)
        return svd.transform(vector_space), svd.components_

    # def PCA(self, vector_space, k):
    #     # implementing PCA using svd implementation si nce scipy PCA doesn't have a sparse
    #     # implementation. The vector space is centered around the mean. Applying a transpose
    #     # over the centered vector space would be like applying SVD on the covariance matrix.
    #     # since cov = (X-mu).T * (X-mu).
    #     mean = vector_space.mean(0)
    #     sparse_representation_centered = csr_matrix(vector_space[0])
    #     for rowNum in range(1, np.size(vector_space, 0)):
    #         sparse_representation_centered = vstack([sparse_representation_centered,
    #                                                  csr_matrix(vector_space[rowNum] - mean)])
    #     # VT contains the contribution of original features to the latent semantics
    #     U, Sigma, VT = randomized_svd(sparse_representation_centered.transpose(),
    #                                   n_components=int(k),
    #                                   n_iter=5,
    #                                   random_state=None)
    #     np.array(U.transpose)
    #     return vector_space.dot(U), VT

    def PCA(self, vector_space, k):
        """ Perform PCA on a given Object Feature matrix

            Parameters
             ----------
            vector_space : Original Object Feature Martix
            k : int
            Number of latent semantics to be which matrix has to be reduced(given by user)

            Returns
            -------
            pca.transform(vector_space) : Reduced Object Feature matrix
            pca.components_ : k latent semantics and old Feature matrix
        """
        pca = PCA(n_components=int(k))
        pca.fit(vector_space)
        return pca.transform(vector_space), pca.components_

    def reduce_dimensions(self, reduction_model, input_param, model, k):
        """ Function to perform dimesion reduction of given

            Parameters
            ----------
            reduction_model :
            input_param :
            model : str
                 text model to use (TF, DF, TF_IDF)
            k : int
                Num of latent semantics to reduce to

            Returns
            -------
            reduced_dimensions
            projection_direction
            object_index_dict : Dict
        """
        vector_space, object_index_dict, term_index_dict = self.get_vector_space(input_param, model)
        # using a sparse implementation to avoid memory error.
        # sparse_representation = csr_matrix(np.array(vector_space[0]))
        # stacking each entities one by one to avoid memory error.
        # for row in range(1, len(vector_space)):
        #    sparse_representation = vstack([sparse_representation, csr_matrix(np.array(vector_space[row]))])
        (reduced_dimensions, projection_direction) = self.reduction_method[reduction_model](vector_space, k)
        GeneralUtils.display_topics(projection_direction, term_index_dict)
        return reduced_dimensions, projection_direction, object_index_dict
