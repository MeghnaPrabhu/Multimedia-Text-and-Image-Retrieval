import pandas as pd
from dbUtils import DBUtils
from scipy.spatial import distance
import math;
from vis_similarity_objects.imgSimliarity import ImgSimilarity
from vis_similarity_objects.locationSimilarityByModel import LocationSimilarityByModel
from vis_similarity_objects.visModelInfo import VisModelInfo
from vis_similarity_objects.totalLocationSimilarity import TotalLocationSimilarity

# Processes the csv files for visual description
class CsvProcessor:
    def __init__(self, base_path, db_operations):
        self._base_path = base_path;
        self._db_operations = db_operations


    def get_data_frame(self, file_name):
        return pd.read_csv(file_name, header=None)

    # normalise using min max
    def normalise_data_frame(self, df):
        col_len = len(df.columns);

        for i in range(1, col_len-1):
            df[i] = (df[i] - df[i].min())/(df[i].max() - df[i].min())
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

    # normalise data frame using min-max normalisation
    def get_normalised_data_frame(self, file_name):
        return self.normalise_data_frame(self.get_data_frame(file_name))

    # get similar locations by comparing all models(uses manhattan distance)
    def find_similar_locations_for_all_models(self, location_id, k):
        loc_id_key_map = DBUtils.create_location_id_key_map(self._db_operations)
        location_key = loc_id_key_map[location_id]
        primary_data_frames_by_model = {}

        most_similar_locations = []
        models = self.get_visual_model_types()

        for model in models:
            file_name = self.get_file_name_from_input(location_key, model.name)
            primary_data_frames_by_model[model.name] = self.get_normalised_data_frame(file_name)

        for id in loc_id_key_map:
            arg_loc_key = loc_id_key_map[id]
            loc_similarity_for_models = []
            total_distance_for_all_models = 0
            if id != location_id:
                for model in models:
                    primary_df = primary_data_frames_by_model[model.name]
                    arg_data_frame = self.get_normalised_data_frame(self.get_file_name_from_input(arg_loc_key, model.name))
                    loc_similarity_for_model = self.get_distance_measure_for_data_frames(arg_loc_key, primary_df, arg_data_frame)
                    loc_similarity_for_model.model = model.name
                    loc_similarity_for_models.append(loc_similarity_for_model)
                    similarity_contribution = loc_similarity_for_model.weighted_distance/model.dimensions;
                    total_distance_for_all_models += similarity_contribution

                most_similar_locations_count = len(most_similar_locations)

                if most_similar_locations_count < k:
                    most_similar_locations.append(TotalLocationSimilarity(arg_loc_key, total_distance_for_all_models,
                                                                          loc_similarity_for_models))
                    most_similar_locations = sorted(most_similar_locations, key=lambda location: location.distance)
                elif most_similar_locations[k - 1].distance > total_distance_for_all_models:
                    most_similar_locations = most_similar_locations[:k - 1]
                    most_similar_locations.append(TotalLocationSimilarity(arg_loc_key, total_distance_for_all_models,
                                                                          loc_similarity_for_models))
                    most_similar_locations = sorted(most_similar_locations, key=lambda location: location.distance)


        print("Input location is {0}".format(location_key))
        self.display_result_for_all_models(most_similar_locations);
        return None

    # Gets the input file based on location key and model
    def get_file_name_from_input(self, location_key, model):
        return "{0}/{1} {2}.csv".format(self._base_path, location_key, model)

    # Calculate Euclidean distance (not used anymore)
    def get_euclidean_distance(self, vector1, vector2):
        vector_len = len(vector1)
        d = 0
        for i in range(0, vector_len):
            diff = vector1[i]-vector2[i]
            d += diff * diff
        return math.sqrt(d)

    # Calculate manhattan dist  (not used anymore)
    def get_manhattan_distance(self, vector1, vector2):
        vector_len = len(vector1)
        d = 0
        for i in range(0, vector_len):
            diff = vector1[i]-vector2[i]
            d += abs(diff)
        return d

    # Display output for a given model
    def display_result_for_model(self, similar_locations):
        print("Most similar locations are")
        for location in similar_locations:
            print("{0} : Distance is {1}".format(location.loc_key, location.weighted_distance))
            print("Most similar images :")
            for image_pair in location.top_similar_images: print("{0} {1} distance : {2}".format(int(image_pair.img1),
                                                                                                 int(image_pair.img2),
                                                                                                 image_pair.distance))
    # displays the output for Task5
    def display_result_for_all_models(self, similar_locations):
        print("Most similar locations are")
        for location in similar_locations:
            print("{0} : Distance is {1}".format(location.location, location.distance))
            print("Contributions from each model :")
            for model in location.contributing_similarity_measures: print("{0} distance : {1}".format(model.model,
                                                                                                      model.weighted_distance))
            print("\n")

    #Gets a concatenated data frame for all locations for a model
    def create_concatenated_and_normalised_data_frame_for_model(self, model, normalise=False):
        loc_id_key_map = DBUtils.create_location_id_key_map(self._db_operations)
        primary_df = None
        for id in loc_id_key_map:
            loc_key = loc_id_key_map[id]
            file_name = self.get_file_name_from_input(loc_key, model)
            if primary_df is None:
                primary_df = self.get_data_frame(file_name)
                primary_df.insert(1, "location", value=id)
            else:
                data_frame_to_add = self.get_data_frame(file_name)
                data_frame_to_add.insert(1, "location", value=id)
                primary_df = pd.concat([primary_df, data_frame_to_add], axis=0,ignore_index=True, sort=False)
        return primary_df if not normalise else self.normalise_data_frame(primary_df)

    # gets the manhattan distance for data frames
    def get_distance_measure_for_data_frames(self,loc_key, primary_df, arg_df):
        distance_matrix = distance.cdist(primary_df.iloc[:, 2:], arg_df.iloc[:, 2:], metric='cityblock')
        total_distance = 0
        col_size = distance_matrix.shape[1]
        for i in range(0, col_size):
            total_distance += distance_matrix[:, i].sum()

        return LocationSimilarityByModel(loc_key, total_distance/col_size,None)

    # calculates euclidean distance and top similar image pairs for 2 data frames
    def get_distance_measure_and_similarity_for_data_frames(self, loc_key, primary_df, arg_df):
        distance_matrix = distance.cdist(primary_df.iloc[:, 2:], arg_df.iloc[:, 2:])
        total_distance = 0
        row_count = len(arg_df.index)
        most_similar_images = []
        for i in range(0, distance_matrix.shape[1]):
            total_distance += distance_matrix[:, i].sum()
            least_distant_3_of_col = distance_matrix[:, i].argsort()[:3]
            least_dist_index_of_3 = least_distant_3_of_col[0]

            if not most_similar_images:

                for image_index in range(0,3):
                    parent_location_image_id = primary_df.iloc[least_distant_3_of_col[image_index], 0]
                    arg_location_image_id = arg_df.iloc[i, 0]
                    dist = distance_matrix[least_distant_3_of_col[image_index]][i]
                    most_similar_images.append(ImgSimilarity(parent_location_image_id, arg_location_image_id, dist))

                most_similar_images = sorted(most_similar_images,
                                             key=lambda image_pair: image_pair.distance)
            elif distance_matrix[least_dist_index_of_3][i] < most_similar_images[2].distance:
                for image_index in range(0, 3):
                    parent_location_image_id = primary_df.iloc[least_distant_3_of_col[image_index], 0]
                    arg_location_image_id = arg_df.iloc[i, 0]
                    dist = distance_matrix[least_distant_3_of_col[image_index]][i]
                    most_similar_images.append(ImgSimilarity(parent_location_image_id, arg_location_image_id, dist))

                most_similar_images = sorted(most_similar_images, key=lambda image_pair: image_pair.distance)
                most_similar_images = most_similar_images[:3]


        weighted_location_distance = total_distance / row_count
        return LocationSimilarityByModel(loc_key, weighted_location_distance, most_similar_images)

    # Gets similar locations and top image pairs for given model and location id
    def find_similar_locations_for_given_model(self, location_id, k, model):
        loc_id_key_map = DBUtils.create_location_id_key_map(self._db_operations)
        location_key = loc_id_key_map[location_id]
        primary_df = self.create_concatenated_and_normalised_data_frame_for_model(model, True)
        most_similar_locations = []
        primary_loc_data_frame = primary_df[primary_df["location"] == location_id]

        for id in loc_id_key_map:
            if id != location_id:
                arg_loc_key = loc_id_key_map[id]
                arg_data_frame = primary_df[primary_df["location"] == id]

                loc_similarity = self.get_distance_measure_and_similarity_for_data_frames(arg_loc_key, primary_loc_data_frame, arg_data_frame)

                most_similar_locations_len = len(most_similar_locations)

                if most_similar_locations_len < k:
                    most_similar_locations.append(loc_similarity)
                    most_similar_locations = sorted(most_similar_locations,
                           key=lambda location: location.weighted_distance)
                elif most_similar_locations[k-1].weighted_distance > loc_similarity.weighted_distance:
                    most_similar_locations = most_similar_locations[:k-1]
                    most_similar_locations.append(loc_similarity)
                    most_similar_locations = sorted(most_similar_locations,
                           key=lambda location: location.weighted_distance)

        print("Input location is {0}".format(location_key))
        self.display_result_for_model(most_similar_locations)


    #Get training data for task6
    def get_training_data(self,training_path):
        training_data = self.get_data_frame(training_path)
        return training_data




