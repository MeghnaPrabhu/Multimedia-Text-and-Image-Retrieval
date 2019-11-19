from dbUtils import DBUtils
from image_viewer.image_viewer_main import ImageViewerMain
class Visualizer:
    def __init__(self, base_path, database_ops):
        self.img_path = base_path + "/img/"
        self.database_ops = database_ops
        self.format = "jpg"

    # Don't change this for custom use cases
    def prepare_file_list(self, image_indexes, obj_index):
        loc_id_key_map = DBUtils.create_location_id_key_map(self.database_ops)
        file_list = []
        for image_index in image_indexes:
            image_tuple = obj_index.iloc[image_index]
            location_id = image_tuple["location"]
            location_key = loc_id_key_map[location_id]
            image_id = image_tuple[0]
            file_list.append(self.img_path + location_key + "/" + str(image_id) + "." + self.format)
        return file_list

    def visualize(self, image_indexes, obj_index):
        image_list = self.prepare_file_list(image_indexes, obj_index)
        image_viewer = ImageViewerMain()
        image_viewer.start_image_viewer(image_list)

    def visualize_with_ids(self, image_id_loc):
        loc_id_key_map = DBUtils.create_location_id_key_map(self.database_ops)
        image_list = []
        for i in image_id_loc:
            location_key = loc_id_key_map[i['loc']]
            image_list.append(self.img_path + location_key + "/" + str(i['imageId']) + "." + self.format)
        image_viewer = ImageViewerMain()
        image_viewer.start_image_viewer(image_list)
