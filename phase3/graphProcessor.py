import numpy as np
from scipy.spatial import distance

from phase1.csvProcessor import CsvProcessor


class GraphProcessor:
    def __init__(self, base_path, databas_ops):
        suffix_image_dir = "/descvis/img"
        self.csvProcessor = CsvProcessor(base_path + suffix_image_dir, databas_ops)

    def get_weighted_image_similarity_graph(self, k):
        image_df = self.csvProcessor.create_concatenated_and_normalised_data_frame_for_model("CM")
        # image_df = image_df.groupby("location").head(100)  # for testing
        distance_matrix = distance.cdist(image_df.iloc[:, 2:], image_df.iloc[:, 2:])
        resultant_matrix = [[-1 for j in range(0, len(distance_matrix))] for i in range(0, len(distance_matrix))]
        row_index = 0
        for row in distance_matrix:
            adjacent_items = [arg for arg in np.argsort(row)[1:k + 1]]
            for adjacent_item in adjacent_items:
                resultant_matrix[row_index][adjacent_item] = row[adjacent_item]
            row_index += 1

        return image_df.iloc[:, 0:2], resultant_matrix

    def get_image_similarity_graph(self, k):
        image_df = self.csvProcessor.create_concatenated_and_normalised_data_frame_for_model("CM")
        distance_matrix = distance.cdist(image_df.iloc[:, 2:], image_df.iloc[:, 2:])
        resultant_matrix = []
        for row in distance_matrix:
            adjacent_items = [arg for arg in np.argsort(row)[1:k + 1]]
            resultant_matrix.append(adjacent_items)
        return image_df.iloc[:, 0:2], resultant_matrix

    def get_transition_matrix_for_weighted(self, sim_graph):
        damping_factor = 0.85
        total_nodes = len(sim_graph)
        initial_value = (1 - damping_factor) / total_nodes
        transition_matrix = [[initial_value for j in range(0, total_nodes)] for i in range(0, total_nodes)]
        index = 0
        while index < total_nodes:
            adjacent_indexes = []
            adjacent_index = 0
            for adjacent_item in sim_graph[index]:
                if adjacent_item != -1:
                    adjacent_indexes.append(adjacent_index)
                adjacent_index += 1
            adjacent_items_len = len(adjacent_indexes)
            for item in adjacent_indexes:
                transition_matrix[index][item] += damping_factor * 1 / adjacent_items_len
            index += 1
        return np.array(transition_matrix)

    def get_transition_matrix(self, sim_graph):
        damping_factor = 0.85
        total_nodes = len(sim_graph)
        initial_value = 1 / total_nodes
        transition_matrix = [[initial_value for j in range(0, total_nodes)] for i in range(0, total_nodes)]
        index = 0

        for row in sim_graph:
            edge_indexes = np.nonzero(row > -1)[0]
            adjacent_item_len = len(edge_indexes)
            for adjacent_item in edge_indexes:
                transition_matrix[index][adjacent_item] += 1/adjacent_item_len
            index += 1
        return np.array(transition_matrix)


    #Using this presently
    def get_page_rank_without_transition_matrix(self, sim_graph, k):
        np_graph = np.array(sim_graph)
        graph_transpose = np_graph.transpose()
        damping_factor = 0.85
        total_nodes = len(sim_graph)
        tolerance = 1.0e-5
        error = 1
        iteration = 0
        # page_rank_current = np.full(shape=(total_nodes, total_nodes), fill_value=(1 - damping_factor) / total_nodes)
        # initial_page_rank = np.full(shape=(total_nodes, total_nodes), fill_value=(1 - damping_factor) / total_nodes)


        page_rank_current = np.array([(1 - damping_factor) / total_nodes for i in range(0, total_nodes)])
        initial_page_rank = np.array([(1 - damping_factor) / total_nodes for i in range(0, total_nodes)])


        while error > tolerance:
            index = 0
            for row in graph_transpose:
                edge_indexes = np.nonzero(row > -1)[0]
                for edge_index in edge_indexes:
                    page_rank_current[index] += initial_page_rank[edge_index] * damping_factor / k
                index += 1
            # page_rank_sum = np.sum(page_rank_current)
            # page_rank_current = page_rank_current/page_rank_sum
            error = np.linalg.norm(page_rank_current - initial_page_rank, 2)
            initial_page_rank = page_rank_current
            page_rank_current = np.array([(1 - damping_factor) / total_nodes for i in range(0, total_nodes)])
            iteration += 1

        return np.argsort(initial_page_rank)[::-1]

        # Using this presently

    def get_personalized_page_rank_general(self, sim_graph, k, seed_values, tolerance=1.0e-5):
        page_rank = self.get_personalized_page_rank_without_transition_matrix(sim_graph, k, seed_values, tolerance)
        return np.argsort(page_rank)[::-1]

    def get_personalized_page_rank_without_transition_matrix(self, sim_graph, k, seed_values, tolerance=1.0e-5):
        np_graph = np.array(sim_graph)
        graph_transpose = np_graph.transpose()
        damping_factor = 0.85
        total_nodes = len(sim_graph)
        error = 1
        iteration = 0
        # page_rank_current = np.full(shape=(total_nodes, total_nodes), fill_value=(1 - damping_factor) / total_nodes)
        # initial_page_rank = np.full(shape=(total_nodes, total_nodes), fill_value=(1 - damping_factor) / total_nodes)

        page_rank_current = np.array([0.0 for i in range(0, total_nodes)])
        initial_page_rank = np.array([0.0 for i in range(0, total_nodes)])
        self.set_seed_values(page_rank_current, seed_values, damping_factor)
        self.set_seed_values(initial_page_rank, seed_values, damping_factor)

        while error > tolerance:
            index = 0
            for row in graph_transpose:
                edge_indexes = np.nonzero(row > -1)[0]
                for edge_index in edge_indexes:
                    page_rank_current[index] += initial_page_rank[edge_index] * damping_factor / k
                index += 1
            # page_rank_sum = np.sum(page_rank_current)
            # page_rank_current = page_rank_current/page_rank_sum
            error = np.linalg.norm(page_rank_current - initial_page_rank, 2)
            initial_page_rank = page_rank_current
            page_rank_current = np.array([0.0 for i in range(0, total_nodes)])
            iteration += 1

        return initial_page_rank

    def set_seed_values(self, rank_array, seed_values, damping_factor):
        for seed_index in seed_values:
            rank_array.itemset(seed_index, (1 - damping_factor) / 3)

    def get_page_rank(self, sim_graph):
        transition_matrix = self.get_transition_matrix(sim_graph)
        pages_len = len(sim_graph)
        intial_page_rank = [1.0 for i in range(0, pages_len)]
        initial_page_rank_transpose = np.array(intial_page_rank).transpose()
        for i in range(0, 100):
            initial_page_rank_transpose = np.matmul(transition_matrix, initial_page_rank_transpose)
        # vectors, values = lin.eig(np.array(transition_matrix))
        return np.argsort(initial_page_rank_transpose)[::-1]


    def ppr_classification(self, sim_graph, labels, k):
        page_rank_scores_for_label = {}
        for label in labels:
            page_rank_for_label = self.get_personalized_page_rank_without_transition_matrix(sim_graph, k, labels[label],
                                                                                            tolerance=1.0e-5)
            page_rank_scores_for_label[label] = page_rank_for_label

        return page_rank_scores_for_label
