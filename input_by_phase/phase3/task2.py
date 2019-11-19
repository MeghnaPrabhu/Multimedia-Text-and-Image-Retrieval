import math
import os
import shutil

import numpy as np
import pandas as pd

from distanceUtils import DistanceUtils
from image_viewer.visualizer import Visualizer


class Task2:
    def __init__(self, base_path, database_ops):
        #        self.display_choice = {1: self.visualize_classes, 2: self.print_classes}
        self.visualizer = Visualizer(base_path, database_ops)
        self.base_path = base_path

    def input(self, obj_index, sim_graph, k):
        sim_graph = np.array(sim_graph)
        sim_graph[sim_graph == -1] = 0
        print("Input number of clusters c for graph")
        number_of_clusters = int(input())
        '''distance matrix contains the distance from each image to all the other images'''
        distance_matrix = DistanceUtils.get_dist_matrix(sim_graph, self.base_path, sim_graph.shape[0], k)
        cluster_choice = 'Y'
        while cluster_choice == 'Y' or cluster_choice == 'Y':
            print("Enter the clustering option 1.) K-means 2.) Max a min ")#3.) Hierarchical Clustering(Precomputed)")
            cluster_option = int(input())
            clusters = {}
            cluster_distances = {}
            if cluster_option == 1:
                clusters, cluster_distances = self.get_kmeans_clusters(distance_matrix, number_of_clusters)
            elif cluster_option == 2:
                clusters, cluster_distances = self.get_max_a_min(distance_matrix, number_of_clusters)
            elif cluster_option == 3:
                clusters = self.get_hierarchical_clustering(distance_matrix, number_of_clusters)

            for idx, cluster_key in enumerate(clusters.keys()):
                print("Number of points in cluster " + str(idx + 1) + ": " + str(len(clusters[cluster_key])))

            # displaying output
            cluster_index = 0
            continue_choice = 'Y'
            print("Select 1.) Visualize 2.) Print 3.) Both")
            input_choice = int(input())
            if input_choice == 1 or input_choice == 3:
                while continue_choice == "Y" or continue_choice == "y":
                    cluster_list = clusters[list(clusters.keys())[cluster_index]]
                    cluster_distances_list = cluster_distances[list(clusters.keys())[cluster_index]]
                    #ordering the points based on its closeness to the centroid
                    cluster_df = pd.DataFrame({'clusters': cluster_list, 'distances': cluster_distances_list})
                    cluster_df = cluster_df.sort_values(['distances'])
                    cluster_list = list(cluster_df['clusters'].values)
                    print()
                    self.visualizer.visualize(cluster_list, obj_index)
                    if cluster_index == len(cluster_list):
                        break
                    print("Press Y to show the next cluster.")
                    continue_choice = input()
                    cluster_index = cluster_index + 1

            cluster_index = 0
            if input_choice == 2 or input_choice == 3:
                while (cluster_index != len(clusters.keys())):
                    print()
                    print("Cluster " + str(cluster_index + 1) + " points")
                    cluster_list = clusters[list(clusters.keys())[cluster_index]]
                    cluster_distances_list = cluster_distances[list(clusters.keys())[cluster_index]]
                    # ordering the points based on its closeness to the centroid
                    cluster_df = pd.DataFrame({'clusters': cluster_list, 'distances': cluster_distances_list})
                    cluster_df = cluster_df.sort_values(['distances'])
                    cluster_list = list(cluster_df['clusters'].values)
                    image_list = self.visualizer.prepare_file_list(cluster_list, obj_index)
                    destination = self.base_path + "/cluster" + str(cluster_index + 1)
                    if os.path.isfile(destination):
                        shutil.rmtree(destination)
                    os.makedirs(destination, exist_ok=True)
                    for image in image_list:
                        shutil.copy2(image, destination)
                        print(image[image.rindex("/") + 1: image.rindex(".")])
                    cluster_index = cluster_index + 1
            print("Press Y to enter the next clustering option")
            cluster_choice = input()
    def get_max_a_min(self, dist_graph, c):
        '''
        Implements the max a min algorithm to cluster the nodes into c clusters.
        :param dist_graph: distance graph given by Johnsons Algorithm
        :param c: number of clusters
        :return: clusters
        '''
        centroids = []
        clusters = {}
        inter_cluster_distances = {}

        outliers = self.is_outlier(dist_graph)
        outlier_points = np.argwhere((outliers))
        # choosing the first centroid randomly
        centroids.append(np.random.randint(dist_graph.shape[0]))
        # setting all infinity values to -infinity so that they don't get chosen as the cluster centroids.
        dist_graph[dist_graph == math.inf] = -math.inf
        # Centroid selection
        while len(centroids) != c + 1:
            rows = dist_graph[centroids]
            # write code to remove outliers from the above rows
            maximally_distant_points = np.sum(rows, axis=0)
            maximally_distant_points[outlier_points] = -math.inf
            next_centroid = np.argmax(maximally_distant_points)
            # Choosing next farthest element from the last selected centroid
            i = 1
            # Handling case where next centroid is already picked as a centroid
            while next_centroid in centroids:
                next_centroid = (-maximally_distant_points).argsort()[i]
                i = i + 1
            centroids.append(next_centroid)
        # calculate distances from other nodes to centroid
        centroids = centroids[1:]
        # setting distances that were infinity initially back to infinity.
        dist_graph[dist_graph == -math.inf] = math.inf
        for idx, node in enumerate(dist_graph):
            distance_to_centroids = [node[centroid] for centroid in centroids]
            index = np.argmin([node[centroid] for centroid in centroids])
            if centroids[index] in clusters.keys():
                clusters[centroids[index]].append(idx)
                inter_cluster_distances[centroids[index]].append(distance_to_centroids[index])
            else:
                clusters[centroids[index]] = [idx]
                inter_cluster_distances[centroids[index]] = [distance_to_centroids[index]]
        return clusters, inter_cluster_distances

    def xget_kmeans_clusters(self, dist_graph, c):
        '''
        implements the kmeans algorithm to find c clusters
        :param dist_graph: node to node distance matrix
        :param c: number of clusters
        :return: clusters
        '''
        outliers = self.is_outlier(dist_graph)
        # sim_graph = np.delete(dist_graph, np.argwhere((points) == True), axis=0)
        # dist_graph = np.delete(dist_graph, np.argwhere((outliers) == True), axis=1)

        clusters = {}
        num_iterations = 100
        # randomly choosing c centroids
        centroids = [np.random.randint(dist_graph.shape[0]) for i in range(c)]

        # removing outliers if they are chosen as centoids
        for idx, centroid in enumerate(centroids):
            if outliers[centroid] == True:
                rand_point = centroid
                while (outliers[rand_point] == True):
                    rand_point = np.random.randint(dist_graph.shape[0])
                centroids[idx] = rand_point

        for iteration in range(num_iterations):
            clusters = {}
            cluster_distances = {}  # represents the distance of the points from the cluster centroid
            # assign points to clusters
            for idx, node in enumerate(dist_graph):
                if outliers[idx]:
                    continue
                centroid_index = np.argmin([node[centroid] for centroid in centroids])
                centroid_dist = np.min([node[centroid] for centroid in centroids])
                if centroids[centroid_index] in clusters.keys():
                    clusters[centroids[centroid_index]].append(idx)
                    cluster_distances[centroids[centroid_index]].append(centroid_dist)
                else:
                    clusters[centroids[centroid_index]] = [idx]
                    cluster_distances[centroids[centroid_index]] = [centroid_dist]
            # updating cluster centroids
            new_centroids = []
            for centroid in centroids:
                cluster_points = clusters[centroid]
                cluster_centre = centroid
                point_centrality_measure = math.inf
                for cluster_point in cluster_points:
                    node = dist_graph[cluster_point]
                    if np.sum(node[cluster_points]) < point_centrality_measure:
                        point_centrality_measure = np.sum(node[cluster_points])
                        cluster_centre = cluster_point

                new_centroid = cluster_centre
                new_centroids.append(new_centroid)

            # check for convergence
            if centroids == new_centroids:
                break;
            else:
                centroids = new_centroids
            # print(iteration)

        '''Classifying outliers'''
        for outlier_index in np.argwhere((outliers) == True):
            nearest_centroid = 0
            nearest_centroid_dist = math.inf
            for centroid_index in centroids:
                if nearest_centroid_dist > dist_graph[outlier_index, centroid_index]:
                    nearest_centroid = centroid_index
                    nearest_centroid_dist = dist_graph[outlier_index, centroid_index]
            clusters[nearest_centroid].append(int(outlier_index))
            cluster_distances[nearest_centroid].append(int(nearest_centroid_dist))

        return clusters, cluster_distances

    def get_hierarchical_clustering(self, sim_graph, c):
        outliers = self.is_outlier(sim_graph)
        # sim_graph = np.delete(sim_graph, np.argwhere((points) == True), axis=0)
        # sim_graph = np.delete(sim_graph, np.argwhere((points) == True), axis=1)

        outlier_points = np.argwhere((outliers))
        sim_graph[outlier_points] = math.inf
        sim_graph[:, outlier_points] = math.inf

        current_number_of_clusters = sim_graph.shape[0]
        sim_graph = np.maximum(sim_graph, sim_graph.T)
        sim_graph[sim_graph == 0] = math.inf
        clusters = {}
        for point in range(sim_graph.shape[0]):
            clusters[point] = [point]

        while current_number_of_clusters != c:
            # x and y index of the minimum element in the matrix
            x_index, y_index = np.unravel_index(sim_graph.argmin(), sim_graph.shape)
            if clusters.get(x_index, False):
                if clusters.get(y_index, False):
                    clusters[x_index].extend(clusters[y_index])
                    clusters.pop(y_index, None)
                else:
                    clusters[x_index].append(y_index)
            elif clusters.get(y_index, False):
                clusters[x_index] = [x_index]
                clusters[x_index].extend(clusters[y_index])
                clusters.pop(y_index, None)
            else:
                clusters[x_index] = [x_index, y_index]
            merged_row = np.minimum(sim_graph[x_index], sim_graph[y_index])
            merged_row[x_index] = math.inf
            merged_row[y_index] = math.inf

            sim_graph[x_index] = merged_row
            sim_graph[:, x_index] = merged_row

            sim_graph[y_index] = math.inf
            sim_graph[:, y_index] = math.inf

            current_number_of_clusters = len(clusters)
            # if current_number_of_clusters % 100 == 0:
            #     print()
            #
            # print(current_number_of_clusters)
        '''Classifying outliers'''
        # for outlier_index in np.argwhere((outliers) == True):
        #     nearest_centroid = 0
        #     nearest_centroid_dist = math.inf
        #     for centroid_index in centroids:
        #         if nearest_centroid_dist > dist_graph[outlier_index, centroid_index]:
        #             nearest_centroid = centroid_index
        #             nearest_centroid_dist = dist_graph[outlier_index, centroid_index]
        #     clusters[nearest_centroid].append(int(outlier_index))

        return clusters

    def is_outlier(self, points, thresh=3.5):
        """
        Returns a boolean array with True if points are outliers and False
        otherwise.

        Parameters:
        -----------
            points : An numobservations by numdimensions array of observations
            thresh : The modified z-score to use as a threshold. Observations with
                a modified z-score (based on the median absolute deviation) greater
                than this value will be classified as outliers.

        Returns:
        --------
            mask : A numobservations-length boolean array.

        References:
        ----------
            Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
            Handle Outliers", The ASQC Basic References in Quality Control:
            Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
        """
        if len(points.shape) == 1:
            points = points[:, None]
        median = np.median(points, axis=0)
        diff = np.sum((points - median) ** 2, axis=-1)
        diff = np.sqrt(diff)
        med_abs_deviation = np.median(diff)

        modified_z_score = 0.6745 * diff / med_abs_deviation

        return modified_z_score > thresh
