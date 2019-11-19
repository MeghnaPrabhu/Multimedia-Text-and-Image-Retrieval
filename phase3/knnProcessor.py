import numpy as np
import pandas as pd
from scipy.spatial import distance

from phase1.csvProcessor import CsvProcessor


class knnProcessor:
    def __init__(self, base_path, databas_ops):
        self.training_path = "/testdatafortask6.csv"
        suffix_image_dir = "/descvis/img"
        self.base_path=base_path
        self.csvProcessor =  CsvProcessor(base_path + suffix_image_dir , databas_ops)


    def find_k_neighbors(self,image_df,training_df,k):
     distance_matrix = distance.cdist(image_df.iloc[:, 2:], training_df.iloc[:, 2:-1],'cityblock')
     resultant_matrix = []
     for row in distance_matrix:
      adjacent_items = [arg for arg in np.argsort(row)[:k]]
      resultant_matrix.append(adjacent_items)
     return image_df.iloc[:, 0:2], resultant_matrix


    def get_knn(self,k):
        image_df = self.csvProcessor.create_concatenated_and_normalised_data_frame_for_model("CM")
        image_df=image_df.rename(columns={ image_df.columns[0]: "imageid" })
        #image_df = image_df.sort_values(['imageid'], ascending=[True])
        training_data= pd.DataFrame(self.csvProcessor.get_training_data(self.base_path+self.training_path))
        training_data.columns=['imageid','label']
        training_df=image_df.merge(training_data, on=['imageid'], how='inner')
        #training_df = training_df.sort_values(['imageid'], ascending=[True])
        obj_index, resultant_matrix=self.find_k_neighbors(image_df,training_df,k)
        return obj_index,resultant_matrix,training_df

    def get_knn_labels(self,knn_graph,image_df,training_df):
        training_df.index=training_df['imageid']
        k_label_df=None
        knn_graph.reset_index(drop=True,inplace=True)
        for x in range(len(knn_graph)):
         n_image_list=list(knn_graph.iloc[x,1:])
         n_label_list=pd.DataFrame(training_df[training_df['imageid'].isin(n_image_list)]['label'])
         n_labels_count=n_label_list.groupby(['label']).size().reset_index().rename(columns={0:'count'}).sort_values(['count'],ascending=False)
         n_labels_count.reset_index(drop=True,inplace=True)
         idx = n_labels_count['count'].max()== n_labels_count['count']
         n_label_max = n_labels_count[idx]
         n_label_max.insert(2, "imageId", value=int(knn_graph.iloc[x, 0]))
         image_index = (image_df.index[image_df['imageid'] == knn_graph.iloc[x, 0]].tolist())
         n_label_max.insert(3, "imageindex", value=int(image_index[0]))
         k_label_df = pd.concat([k_label_df, n_label_max], axis=0)

        k_label_df.columns = ['label', 'count', 'image', 'imageindex']
        class_dict = k_label_df.groupby('label')[['imageindex', 'image']].apply(lambda g: g.values.tolist()).to_dict()
        # class_dict={k: list(v) for k,v in k_label_df.groupby("label")[["imageindex","image"]]}
        k_label_df.to_csv('final_Label.csv')
        pd.DataFrame([class_dict], columns=class_dict.keys()).to_csv('Label_dict.csv')
        return (k_label_df, class_dict)
