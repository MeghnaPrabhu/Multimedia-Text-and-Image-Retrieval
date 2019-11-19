import numpy as np
import pandas as pd


class GeneralUtils:
    @staticmethod
    def get_conversion_func_for_text_model(model):
        def convert_int(item):
            return int(item)

        return_func = convert_int
        if model == "TF_IDF":
            def convert_float(item):
                return float(item)

            return_func = convert_float

        return return_func

    # Display latent semantics and composition; Prints only top 10 terms for each latent semantic
    @staticmethod
    def display_topics(components, term_list):
        for topic_idx, topic in enumerate(components):
            print("\nTopic {0}:".format(topic_idx))
            res = [term_list[i] + ":" + str(topic[i]) + "\n" for i in topic.argsort()[::-1]]


    @staticmethod
    def display_graph(obj_index, sim_graph):
        node_index = 0
        nodes_total = len(sim_graph)
        np_graph = np.array(sim_graph)
        for row in np_graph:
            print(str(obj_index.iloc[node_index, 0]) + ":")
            edge_indexes = np.nonzero(row > -1)[0]
            print(",".join(str(obj_index.iloc[i, 0]) for i in edge_indexes))

    @staticmethod
    def get_graph(obj_index, sim_graph,training_data):
        node_index = 0
        nodes_total = len(sim_graph)
        obj_df=None
        result=None
        while node_index < nodes_total:
           obj_df = pd.DataFrame([obj_index.iloc[node_index, 0]])
           knn_series = (",".join(str(training_data.iloc[i, 0]) for i in sim_graph[node_index]))
           knn_list = pd.DataFrame([knn_series.split(",")])
           result = pd.concat([result, pd.concat([obj_df, knn_list],ignore_index=True, axis=1)], axis=0, sort=False)
           node_index += 1
        return(result)



    @staticmethod
    def get_projection(vector, projection_direction):
        # function that projects one vector onto the other.
        # Args:
        #   @vector: vector to be projected
        #   @projection_direction: vector on which you want it to be projected
        # Returns:
        #   projected results as np array.
        return np.dot(vector, projection_direction.T)
