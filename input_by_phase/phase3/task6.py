from generalUtils import GeneralUtils
from image_viewer.visualizer import Visualizer
from phase3.graphProcessor import GraphProcessor
from phase3.knnProcessor import knnProcessor


class Task6:
    def __init__(self, base_path, database_ops):
        self.knnProcessor = knnProcessor(base_path, database_ops)
        self.graph_processor = GraphProcessor(base_path, database_ops)
        self.input_choice = {1: self.knn_classification, 2: self.ppr_classification}
        self.classification_file_path = base_path + "/classification_input.txt"
        self.visualizer = Visualizer(base_path, database_ops)
        self.display_choice = {1: self.visualize_classes, 2: self.print_classes}

    def read_file(self):
        labels = []
        with open(self.classification_file_path, 'r') as f:
            content = f.readlines()
            for line in content:
                split_line = line.split()
                labels.append([split_line[0], split_line[1]])
        return labels

    def knn_classification(self):
        print("Enter value of k")
        k = int(input())
        obj_index, knn_images,training_df=self.knnProcessor.get_knn(k)
        knn_graph= GeneralUtils.get_graph(obj_index, knn_images,training_df)
        training_df.to_csv('train.csv')
        knn_graph.to_csv('neighbors.csv')
        knn_labels, knn_dict = self.knnProcessor.get_knn_labels(knn_graph, obj_index, training_df)
        total_nodes = len(knn_graph)
        continue_choice = "Y"
        while continue_choice != "N":
            print("Select 1. Visualize 2. Print")
            input_choice = int(input())
            self.display_choice[input_choice](knn_dict, obj_index, total_nodes)
            print("Press N to stop displaying results")
            continue_choice = input()

    def partition_labels_into_seed_values(self, labels):
        seed_dict = {}
        for label in labels:
            if label[1] not in seed_dict:
                seed_dict[label[1]] = []
            seed_dict[label[1]].append(label[0])

        return seed_dict

    def convert_labels_to_index(self, labels_dict, obj_index):
        label_obj_index_dict = {}
        for label in labels_dict:
            seed_nodes = labels_dict[label]
            seed_indexes = obj_index[obj_index[0].isin(seed_nodes) == True].index.values
            label_obj_index_dict[label] = seed_indexes
        return label_obj_index_dict

    def pre_process_ppr(self, obj_index):
        labels = self.read_file()
        labels_dict = self.partition_labels_into_seed_values(labels)
        labels_obj_index_dict = self.convert_labels_to_index(labels_dict, obj_index)
        return labels_obj_index_dict

    def visualize_classes(self, classified_items, obj_index, total_nodes):
        classified_items = self.customize_classified_items_list(classified_items, add_score=False)
        label_choice = {}
        count = 1
        print("Select label ")
        for label in classified_items:
            print(str(count) + " " + label)
            label_choice[count] = label
            count += 1

        input_label_choice = int(input())
        selected_label = label_choice[input_label_choice]
        image_index_list = classified_items[selected_label]
        self.visualizer.visualize(image_index_list, obj_index)

    def classify_page_scores(self, page_scores, total_nodes):
        classified_result = {}

        for index in range(0, total_nodes):
            max_label = None
            max_score = 0.0
            for label in page_scores:
                if page_scores[label][index] >= max_score:
                    max_label = label
                    max_score = page_scores[label][index]

            if max_label not in classified_result:
                classified_result[max_label] = []

            classified_result[max_label].append([index, max_score])

        return classified_result

    def customize_classified_items_list(self, classified_result, add_score=False, reverse=False):
        classified_result_without_score = {}
        for key in classified_result:
            val = classified_result[key]
            val.sort(key=lambda x: x[1], reverse=reverse)
            classified_result_without_score[key] = [item[0] for item in val]

        return classified_result if add_score else classified_result_without_score


    def ppr_classification(self):
        print("Input k for graph")
        k = int(input())
        # unweighted graph func also available
        obj_index, sim_graph = self.graph_processor.get_weighted_image_similarity_graph(k)
        labels = self.pre_process_ppr(obj_index)

        page_rank_scores_for_labels = self.graph_processor.ppr_classification(sim_graph, labels, k)
        total_nodes = len(sim_graph)
        classified_items = self.classify_page_scores(page_rank_scores_for_labels, total_nodes)
        continue_choice = "Y"
        while continue_choice != "N":
            print("Select 1. Visualize 2. Print")
            input_choice = int(input())
            self.display_choice[input_choice](classified_items, obj_index, total_nodes)
            print("Press N to stop displaying results")
            continue_choice = input()

    def print_classes(self, classified_items, obj_index, total_nodes):
        classified_items = self.customize_classified_items_list(classified_items,add_score=True, reverse=True)
        for label in classified_items:
            print(label + ": ")
            for item in classified_items[label]:
                print(str(obj_index.iloc[item[0]][0]) + " " + str(item[1]) + " ,")

    def input(self):
        print("Enter choice of classification 1. KNN 2. PPR")
        choice = int(input())
        self.input_choice[choice]()
