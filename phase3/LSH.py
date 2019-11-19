import math
from collections import defaultdict
from functools import reduce
from pprint import pprint

import numpy as np
import pandas as pd

from phase1.csvProcessor import CsvProcessor

SEED = 12
np.random.seed(SEED)
IMAGE_ID_COL = 'imageId'


class LSH:

    def __init__(self, hash_obj, num_layers, num_hash, vec, b, w):
        self.hash_obj = hash_obj
        self.num_layers = num_layers
        self.num_hash = num_hash
        self.vec = vec
        self.b = b
        self.w = w

    def create_hash_table(self, img_vecs, verbose=False):
        """ Vectorized hash function to bucket all img vecs

            Returns
            -------
            hash_table : List of List of defaultdicts


        """
        hash_table = self.init_hash_table()
        for vec in img_vecs:
            img_id, img_vec = vec[0], vec[1:]
            for idx, hash_vec in enumerate(hash_table):
                buckets = self.hash_obj.hash(img_vec, self.vec[idx], self.b[idx], self.w)
                for i in range(len(buckets)):
                    hash_vec[i][buckets[i]].add(img_id)
        # TODO save hashtable somewhere
        if verbose:
            pprint(hash_table)
        return hash_table

    def init_hash_table(self):
        hash_table = []
        for i in range(self.num_layers):
            hash_layer = []
            for j in range(self.num_hash):
                hash_vec = defaultdict(set)
                hash_layer.append(hash_vec)
            hash_table.append(hash_layer)
        return hash_table

    def find_ann(self, query_point, hash_table, k=5):
        candidate_imgs = set()
        num_conjunctions = self.num_hash
        for layer_idx, layer in enumerate(self.vec):
            hash_vec = hash_table[layer_idx]
            buckets = self.hash_obj.hash(query_point, layer, self.b[layer_idx], self.w)
            cand = hash_vec[0][buckets[0]].copy()
            # self.test(hash_vec[1])
            for ix, idx in enumerate(buckets[1:num_conjunctions]):
                # needs ix+1 since we already took care of index 0
                cand = cand.intersection(hash_vec[ix + 1][idx])
            candidate_imgs = candidate_imgs.union(cand)
            if len(candidate_imgs) > 4 * k:
                print(f'Early stopping at layer {layer_idx} found {len(candidate_imgs) }')
                break
        if len(candidate_imgs) < k:
            if num_conjunctions > 1:
                num_conjunctions -= 1
                return self.find_ann(query_point, hash_table, k=k)
            else:
                print('fubar')
        return candidate_imgs

    def post_process_filter(self, query_point, candidates, k):
        distances = [{IMAGE_ID_COL: int(row[IMAGE_ID_COL]),
                      'dist': self.hash_obj.dist(query_point, row.drop(IMAGE_ID_COL))}
                     for idx, row in candidates.iterrows()]
        # distances []
        # for row in candidates.iterrows():
        #    dist = self.hash_obj.dist(query_point, )
        return sorted(distances, key=lambda x: x['dist'])[:k]


class l2DistHash:

    def hash(self, point, vec, b, w):
        """
            Parameters
            ----------
            point :
            vec:

            Returns
            -------
            numpy array of which buckets point falls in given layer
        """
        val = np.dot(vec, point) + b
        val = val * 100
        res = np.floor_divide(val, w)
        return res

    def dist(self, point1, point2):
        v = (point1 - point2)**2
        return math.sqrt(sum(v))


class lshOrchestrator:

    def __init__(self, base_path, databas_ops):
        suffix_image_dir = "/descvis/img"
        self.csvProcessor = CsvProcessor(base_path + suffix_image_dir, databas_ops)

    def run_lsh(self, input_vec, num_layers, num_hash):
        w = 5
        dim = len(input_vec[0])
        vec = np.random.rand(num_layers, num_hash, dim - 1)
        b = np.random.randint(low=0, high=w, size=(num_layers, num_hash))
        l2_dist_obj = l2DistHash()
        lsh = LSH(hash_obj=l2_dist_obj, num_layers=num_layers, num_hash=num_hash, vec=vec, b=b, w=w)

        hashTable = lsh.create_hash_table(input_vec, verbose=False)
        return hashTable

    def get_combined_visual_model(self, models):
        model_dfs = []
        for model_name in models:
            df = self.csvProcessor.create_concatenated_and_normalised_data_frame_for_model(model_name, normalise=True)
            df = df.rename(columns={df.columns[0]: IMAGE_ID_COL})
            model_dfs.append(df)
        img_dfs = reduce(lambda left, right: pd.merge(left, right, on=[IMAGE_ID_COL, 'location']), model_dfs)
        return img_dfs

    def img_ann(self, query, k, num_layers=100, num_hash=30, layer_file_name=None):
        models = ['CN3x3', 'CM3x3', 'HOG', 'CSD', 'GLRLM']
        print(f'Using models : {models}')
        img_df = self.get_combined_visual_model(models)
        img_id_loc_df = img_df[[IMAGE_ID_COL, 'location']]
        img_df.drop('location', axis=1, inplace=True)
        assert img_df.shape[1] > 256
        n, dim = img_df.shape
        # w = int(math.sqrt(n)) if n > 100 else k**3
        w = 400
        # Create vector with rand num in num_layers X num_hash X dim-1(1 dim for img_id)
        vec = np.random.rand(num_layers, num_hash, dim - 1)
        #vec = np.arange(num_layers*num_hash*(dim-1)).reshape(num_layers, num_hash, dim-1)
        b = np.random.randint(low=0, high=w, size=(num_layers, num_hash))
        # b = np.arange(num_layers*num_hash).reshape(num_layers, num_hash)

        l2_dist_obj = l2DistHash()
        lsh = LSH(hash_obj=l2_dist_obj, num_layers=num_layers, num_hash=num_hash, vec=vec, b=b, w=w)
        hash_table = lsh.create_hash_table(img_df.values)
        query_vec = img_df.loc[img_df[IMAGE_ID_COL] == int(query)].drop(IMAGE_ID_COL, axis=1)
        t = query_vec.shape[1]
        query_vec = query_vec.values.reshape(t, )
        candidate_ids = lsh.find_ann(query_point=query_vec, hash_table=hash_table, k=k)
        candidate_vecs = img_df.loc[img_df[IMAGE_ID_COL].isin(candidate_ids)]
        if not candidate_ids:
            return None
        dist_res = lsh.post_process_filter(query_point=query_vec, candidates=candidate_vecs, k=k)
        for i in dist_res:
            img_id = i[IMAGE_ID_COL]
            i['loc'] = img_id_loc_df.loc[img_id_loc_df[IMAGE_ID_COL] == img_id, 'location'].item()
        return dist_res
