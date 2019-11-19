from collections import Counter
from pprint import pprint
import sys
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from pathlib import Path
# from tensorly.decomposition import parafac
from tensorly.decomposition import candecomp_parafac as cp

from phase2.dimReductionEngineForText import DimReductionEngine


class TensorProcessor:

    def __init__(self, db_ops):
        sns.set_style("whitegrid")
        self.base_path = sys.argv[1]
        self.RANDOM_STATE = 12
        self.dim_reduction_engine = DimReductionEngine(db_ops)

    def cp_decomposition(self, tensor, rank):
        """ Perform rank k cp_decomposition on tensor
            Parameters
            ----------
            tensor : tensorly.tensor
            rank : int
                rank of cp decomposition

            Returns
            -------
            factors : list of m factor matrices of tensor with m modes

        """
        factors = cp.parafac(tensor, rank)
        return factors

    def create_k_clusters(self, X, k, n_jobs=-1):
        """ Use k-means algorithm to create k-clusters

            Parameters
            ----------
            X : Array-like
                data matrix for which clustering has to be done
            k : int
                number of clusters to create
            n_jobs : int
                number of cores to use

            Returns
            -------
            kmeans.cluster_centers_ : Array

        """

        kmeans = KMeans(n_clusters=k, random_state=self.RANDOM_STATE, n_jobs=n_jobs).fit(X)
        return kmeans

    def get_factors(self, k, tensor):
        factor_file_name = self.base_path + "/factor_" + str(k)
        tensor_file = Path(factor_file_name + "_0.npy")
        factors = []
        if not tensor_file.is_file():
            factors = self.cp_decomposition(tensor, rank=k)
            for i in range(3):
                np.save(factor_file_name + "_" + str(i), factors[i])
        else:
            for i in range(3):
                factor = np.load(factor_file_name + "_" + str(i) + ".npy")
                factors.append(factor)
        return factors

    def process_tensor(self, tensor, k):
        """ Run task 7 of phase 2

            Parameters
            ----------
            k : int
                Number of k groups to be created, also the rank of cp-decomposition

            Returns
            -------
            res : list[list[dict]]
                res is a list of size 3 (3 modes of tensor) each element is a list of k dicts having info about the k clusters
        """
        print(f'Loaded tensor with shape : {tensor.shape}')

        # factors = self.cp_decomposition(tensor, rank=k)
        factors = self.get_factors(k, tensor)
        factor_modes = [{'id': 'user', 'pos': 111}, {'id': 'image', 'pos': 112}, {'id': 'location', 'pos': 113}]
        # assert len(factors) == 3
        # fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
        ax = []
        for i, factor in enumerate(factors):
            factor_matrix_index = self.dim_reduction_engine.create_index_tensor_factor(factor_modes[i]['id'])
            k_means = self.create_k_clusters(np.nan_to_num(factor), k)
            label_cntr = Counter(k_means.labels_)
            print(f"Tensor partitioned into {k} groups on {factor_modes[i]['id']} mode")
            cluster_centers = k_means.cluster_centers_
            res = []
            for j, center in enumerate(cluster_centers):
                # print(f'{j}: cluster center : {center}; number of points : {label_cntr[j]}')
                sample_points = self.get_k_idx(k_means.labels_, j, factor_matrix_index)
                res.append({'cluster_number': j, 'number_of_points': label_cntr[j],
                            factor_modes[i]['id'] + 's': sample_points})
            pprint(res)
            try :
                t = plt.subplot(1, 1, i+1)
                t = sns.scatterplot(data=factor, hue=k_means.labels_)
                t.set_title(f"Clustered {0}".format(factor_modes[i]['id']))
                ax.append(t)
                #plt.savefig(f'k_means_res.png')
            except Exception as e:
                pass
        return res

    def get_k_idx(self, src_lis, trg, factor_matrix_index, k=10000):
        res = []
        idx = 0
        for x in src_lis:
            if x == trg:
                res.append(factor_matrix_index[idx])
                if k == 1:
                    break
                k -= 1
            idx += 1
        return res
