import torch
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data.dataset import Dataset
from glob import glob

class GazeFollow_body_head(Dataset):
    def __init__(self, data_dir, test=True):
        super(GazeFollow_body_head, self).__init__()
        self.data_dir = data_dir
        csv_path = os.path.join(data_dir, 'test_annotations_release_persondet.txt') if test else os.path.join(data_dir, 'train_annotations_release_persondet.txt')
        if test:
            column_names = ['path', 'idx', 'body_bbox_x', 'body_bbox_y', 'body_bbox_w', 'body_bbox_h', 'eye_x', 'eye_y',
                            'gaze_x', 'gaze_y', 'bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max', 'meta', 'ori_name', 
                            'body_x1', 'body_y1', 'body_x2', 'body_y2']

            df = pd.read_csv(csv_path, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
            df = df.groupby(['path', 'eye_x'])
            self.keys = list(df.groups.keys())
            self.img_paths = list(df.groups.keys())
        else:
            column_names = ['path', 'idx', 'body_bbox_x', 'body_bbox_y', 'body_bbox_w', 'body_bbox_h', 'eye_x', 'eye_y',
                            'gaze_x', 'gaze_y', 'bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max', 'inout', 'meta', 'ori_name',
                            'body_x1', 'body_y1', 'body_x2', 'body_y2']
            df = pd.read_csv(csv_path, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
            df=df[df['inout']!=-1]
            df.reset_index(inplace=True)
            self.img_paths = df['path']
                
        self.test = test
        self.df = df
    def __getitem__(self, index):

        if self.test:
            g = self.df.get_group(self.keys[index])
            row = g.iloc[0]
            inout = 1 
        else:
            row = self.df.iloc[index]            
            inout = row['inout']
        img_name = row['path']
        img_path = os.path.join(self.data_dir, img_name)
        img = Image.open(img_path)
        width, height = img.size
        eye_x = row['eye_x'] 
        path = row['path']
        x_min = row['bbox_x_min']
        y_min = row['bbox_y_min']
        x_max = row['bbox_x_max']
        y_max = row['bbox_y_max']
        
        head_box = np.array([x_min, y_min, x_max,y_max])
        
        body_box = row[['body_x1', 'body_y1', 'body_x2', 'body_y2']].values
        body_box = body_box.astype(int)
        
        detected = 1
        if body_box[0]<0:
            detected=0 
        return img_path, eye_x, body_box, head_box, inout, detected
        
    def __len__(self):
        return len(self.img_paths)
    
    
if __name__=='__main__':
    gf_data_dir = '/nfs/bigcortex.cs.stonybrook.edu/add_disk0/qiaomu/datasets/gaze/gazefollow'
    dataset = GazeFollow_body_head(gf_data_dir)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers = 0, shuffle=False)
    for idx, data in enumerate(dataloader):
        img_path, body_box, head_box = data    
        import pdb
        pdb.set_trace()