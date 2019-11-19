import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation, PCA

from dbUtils import DBUtils
from distanceUtils import DistanceUtils
from generalUtils import GeneralUtils
from vis_similarity_objects.visModelInfo import VisModelInfo


class DimReductionEngineForVisual:
    def __init__(self, base_path, db_operations):
        self._base_path = base_path;
        self._db_operations = db_operations
        self.reduction_method = {1: self.PCA, 2: self.SVD, 3: self.LDA}
        self.data_load_option = {4: self.create_concatenated_and_normalized_data_frame_for_a_location_model,
                                 5: self.create_concatenated_and_normalized_data_frame_for_a_location}
        self.normalise_method = {1: self.normalise_mean_data_frame, 3: self.normalise_data_frame}
        self.normalise_methodformodel = {1: self.normalise_mean_data_frame, 3: self.normalise_data_frame_withCol}


    def get_data_frame(self, file_name):
        return pd.read_csv(file_name, header=None)

    # normalise data frame using min-max normalisation
    def get_normalised_data_frame(self, file_name):
        return self.normalise_data_frame(self.get_data_frame(file_name))

    # normalise using min max for a given dataframe with column range
    def normalise_data_frame_withCol(self, df):
        col_len = len(df.columns);
        for i in range(1, col_len - 1):
            df[i] = (df[i] - df[i].min()) / (df[i].max() - df[i].min())
        return df
    # normalise using min max for a given dataframe
    def normalise_data_frame(self, df):
        x = df.values  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df = pd.DataFrame(x_scaled)
        return df

    # normalise using mean for a given dataframe
    def normalise_mean_data_frame(self, df):
        # dfvector = df.iloc[:,1:] # removed location id and image name
        # df.iloc[:,1:] = dfvector.sub(df.mean(axis=0), axis=1)
        col_len = len(df.columns)
        for i in range(1, col_len - 1):
            df[i] = (df[i].mean() - df[i])
        return df

    # All visual models with dimension count
    def get_visual_model_types(self):
        models = []
        models.append(VisModelInfo("CN", 11))
        models.append(VisModelInfo("HOG", 81))
        models.append(VisModelInfo("CM", 9))
        models.append(VisModelInfo("LBP", 16))
        models.append(VisModelInfo("CSD", 64))
        models.append(VisModelInfo("GLRLM", 44))
        models.append(VisModelInfo("CN3x3", 99))
        models.append(VisModelInfo("CM3x3", 81))
        models.append(VisModelInfo("GLRLM3x3", 396))
        models.append(VisModelInfo("LBP3x3", 144))
        return models

    def get_file_name_from_input(self, location_key, model):
        return "{0}/{1} {2}.csv".format(self._base_path, location_key, model)


    def create_concatenated_and_normalised_data_frame_for_model(self, model,input_option):
        """ Concatenate data frame for all locations for a given model
            Parameters
            ----------
            model : model given by user
            input_option : int
                           Type of reduction algorithm 1.PCA 2.SVD 3.LDA

            Returns
            -------
            primary_df : For PCA and LDA, it returns normalised dataframe
                         For SVD dataframe with all locations for a given model
        """
        loc_id_key_map = DBUtils.create_location_id_key_map(self._db_operations)
        primary_df = None
        for id in loc_id_key_map:
            loc_key = loc_id_key_map[id]
            file_name = self.get_file_name_from_input(loc_key, model)
            if primary_df is None:
                primary_df = self.get_data_frame(file_name)
                primary_df.insert(1, "locationId", value=id)
            else:
                data_frame_to_add = self.get_data_frame(file_name)
                data_frame_to_add.insert(1, "locationId", value=id)
                primary_df = pd.concat([primary_df, data_frame_to_add], axis=0, sort=False)
        return primary_df if input_option ==2 else self.normalise_methodformodel[input_option] (primary_df) #if not SVD .then normalise

    def create_concatenated_and_normalized_data_frame_for_a_location(self, location_id, input_option, model=None):
        loc_id_key_map = DBUtils.create_location_id_key_map(self._db_operations)
        location_key = loc_id_key_map[int(location_id)]
        primary_data_frames_by_model = pd.DataFrame()
        for model in self.get_visual_model_types():
            file_name = self.get_file_name_from_input(location_key, model.name)
            data_frame_to_add = self.get_data_frame(file_name)
            data_frame_to_add.drop(data_frame_to_add.columns[0], axis=1, inplace=True)
            primary_data_frames_by_model = pd.concat([primary_data_frames_by_model, data_frame_to_add], ignore_index= True, axis=1,
                                                     sort=False)
        return primary_data_frames_by_model if input_option == 2 else self.normalise_method[input_option](
            primary_data_frames_by_model)  # if not SVD .then normalise

    def create_concatenated_and_normalized_data_frame_for_a_location_model(self, location_id, input_option, model):
        """ Get data frame for a given location for a given model
            Parameters
            ----------
            model : model given by user
            location_id : int
                           Location id given by user

            Returns
            -------
            data_frame_to_add : Data frame for a given location for a given model
        """
        loc_id_key_map = DBUtils.create_location_id_key_map(self._db_operations)
        location_key = loc_id_key_map[int(location_id)]
        primary_data_frames_by_model = pd.DataFrame()
        file_name = self.get_file_name_from_input(location_key, model)
        data_frame_to_add = self.get_data_frame(file_name)
        data_frame_to_add.drop(data_frame_to_add.columns[0], axis=1, inplace=True)
        return data_frame_to_add if input_option == 2 else self.normalise_method[input_option](
            data_frame_to_add)

    def project_data_onto_new_dimensions(self, target_entity_id, total_target_objects, projection_direction, task,
                                         model, input_param):
        post_projection_vectors = {}
        for i in range(1, total_target_objects + 1):
            if i != int(target_entity_id):
                location_features = self.data_load_option[task]((i), input_param, model)
                projected_vector = GeneralUtils.get_projection(location_features, projection_direction)
                post_projection_vectors[i] = projected_vector
        return post_projection_vectors

    def LDA(self, vector_space, k):
        """ Perform LDA on a given Object Feature matrix
            Parameters
            ----------
            vector_space : Original Object Feature Martix
             k : int
                 Number of latent semantics to be which matrix has to be reduced(given by user)

            Returns
            -------
            res : Reduced Object Feature matrix
            lda.components_ : k latent semantics and old Feature matrix
        """
        vector_space_np = np.array(vector_space)
        lda = LatentDirichletAllocation(n_components=k)
        res = lda.fit_transform(vector_space_np)
        return res, lda.components_

    def SVD(self, vector_space, k):
        """ Perform SVD on a given Object Feature matrix
            Parameters
            ----------
            vector_space : Original Object Feature Martix
             k : int
                 Number of latent semantics to be which matrix has to be reduced(given by user)

            Returns
            -------
            res : Reduced Object Feature matrix
            lda.components_ : k latent semantics and old Feature matrix
        """
        svd = TruncatedSVD(n_components=int(k))
        svd.fit(vector_space.values)
        return svd.transform(vector_space.values), svd.components_

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
        pca.fit(vector_space.values)
        return pca.transform(vector_space), pca.components_

    def reduce_dimensions(self, input_param, data_option, entity_id, k):
        loc_id_key_map = DBUtils.create_location_id_key_map(self._db_operations)
        vector_space = self.data_load_option[data_option](entity_id, input_param)
        (reduced_dimensions, VT) = self.reduction_method[input_param](vector_space, k)
        post_projection_vectors = self.project_data_onto_new_dimensions(entity_id, len(loc_id_key_map), VT,5,None, input_param ) #Model=None
        return reduced_dimensions, post_projection_vectors, loc_id_key_map


    def reduce_dimensions_givenmodel(self,input_option, model, k, count):
          """ Gives 5 related images and location for a given model and image id  after preforming dimensionality reduction
            Parameters
            ----------
            vector_space : Original Object Feature Martix
             input_option: int
                           Reduction algorithm given by the user 1.PCA 2.SVD 3.LDA
             model: model given by user
             k : int
                 Number of latent semantics to be which matrix has to be reduced(given by user)
             count: 5
                  Given in task 3

            Returns
            -------
            Gives 5 related images and 5 related locations for a given model and image id
          """
          loc_id_key_map = DBUtils.create_location_id_key_map(self._db_operations)
          input_method = int(input_option)
          count=int(count)
          vector_space=self.create_concatenated_and_normalised_data_frame_for_model(model,input_method)
          vector_space=vector_space.rename(columns={ vector_space.columns[0]: "image" })
          vector_space = vector_space.sort_values(['locationId','image'], ascending=[True,True])       # sort based on loc and img
          vector_space.reset_index(drop=True, inplace=True)
          (latent_semantics_matrix,VT)=self.reduction_method[input_method](vector_space.iloc[:,2:],k)
          latent_semantics=pd.DataFrame(latent_semantics_matrix)
          print("Latent semantics are")
          print(pd.DataFrame(VT))
          latent_semantics.reset_index(drop=True, inplace=True)
          reduced_space = pd.concat([vector_space.iloc[:,:2], latent_semantics], axis=1)
          print("Enter Image ID to search")
          image_id = int(input())
          (ImageMatrix,LocationMateix)=DistanceUtils.find_similar_images_locations_for_given_model(image_id, k, model, reduced_space,count,loc_id_key_map)
          df_loc_id_key_map = pd.DataFrame(list(loc_id_key_map.items()), columns=['locationId', 'locationKey'])
          ImageMatrix = pd.DataFrame(pd.merge(ImageMatrix,df_loc_id_key_map,on='locationId',how='left'))
          LocationMateix=pd.DataFrame(pd.merge(LocationMateix,df_loc_id_key_map,on='locationId',how='left'))
          print("5 related Images are")
          print(ImageMatrix.loc[:,['image','locationKey','dist']])
          print("5 related locations are")
          print(LocationMateix.loc[:,['locationKey','dist','locationId']])

    def reduce_dimensions_given_Location_Model(self, input_param, model, entity_id, k):
        """ Gives 5 related location for a given model and image id  after preforming dimensionality reduction to k latent semantics
            Parameters
            ----------
            vector_space : Original Object Feature Martix
             input_param: int
                           Reduction algorithm given by the user 1.PCA 2.SVD 3.LDA
             model: model given by user
             k : int
                 Number of latent semantics to be which matrix has to be reduced(given by user)
             entity_id: 5
                  Location id given by the users

            Returns
            -------
            reduced_dimensions, post_projection_vectors, loc_id_key_map
            Gives 5 related  locations for a given model and location id
        """

        loc_id_key_map = DBUtils.create_location_id_key_map(self._db_operations)
        vector_space = self.create_concatenated_and_normalized_data_frame_for_a_location_model(entity_id,
                                                                                               input_param, model
                                                                                               )
        (reduced_dimensions, VT) = self.reduction_method[input_param](vector_space, k)
        post_projection_vectors = self.project_data_onto_new_dimensions(entity_id, len(loc_id_key_map), VT, 4, model,
                                                                        input_param)
        return reduced_dimensions, post_projection_vectors, loc_id_key_map



