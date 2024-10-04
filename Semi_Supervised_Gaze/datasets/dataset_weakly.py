import torch
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import pandas as pd
import pickle, json
import matplotlib as mpl
import traceback
import ast
import re
mpl.use('Agg')
from matplotlib import cm
import matplotlib.pyplot as plt
from skimage.transform import resize
import cv2
import pdb
from utils import imutils
from utils import myutils
from config import *


class GazeFollow_weakly(Dataset):
    def __init__(self, cfg, transform, test=False, ratio=0.1, use_sup_only=False, use_unsup_only=False, no_aug=False):
        data_dir = cfg.gazefollow_base_dir
        self.data_dir = data_dir
        self.no_aug = no_aug
        
        assert not (use_sup_only and use_unsup_only)
        if test:
            csv_path = os.path.join(data_dir, "test_annotations_release_persondet.txt") 
        elif (use_sup_only or use_unsup_only) and ratio!=1.0:
            if use_sup_only:
                csv_path = os.path.join(data_dir,  "weak_supervision", "train_annotations_weak_ratio{}_use.txt".format(ratio))
            else:
                csv_path = os.path.join(data_dir,  "weak_supervision", "train_annotations_weak_ratio{}_left.txt".format(ratio))
                
        else:
            csv_path = os.path.join(data_dir, "train_annotations_release_persondet.txt")
        
        if ratio<1.0:
            unsup_label, sup_label = os.path.join(data_dir,  "weak_supervision", "train_annotations_weak_ratio{}_left.txt".format(ratio)), os.path.join(data_dir,  "weak_supervision", "train_annotations_weak_ratio{}_use.txt".format(ratio))
       
        self.diff_pred_dir, self.baseline_pred_dir = os.path.join(data_dir, "diff_pred"), os.path.join(data_dir, "baseline_pred")
         
        if test:
            column_names = ['path', 'idx', 'body_bbox_x', 'body_bbox_y', 'body_bbox_w', 'body_bbox_h', 'eye_x', 'eye_y',
                            'gaze_x', 'gaze_y', 'bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max', 'meta', 'ori_name',
                            'body_x1', 'body_y1', 'body_x2', 'body_y2']
            
            df = pd.read_csv(csv_path, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
            
            df = df[['path', 'eye_x', 'eye_y', 'gaze_x', 'gaze_y', 'bbox_x_min', 'bbox_y_min', 'bbox_x_max',
                    'bbox_y_max']].groupby(['path', 'eye_x'])
             
            self.keys = list(df.groups.keys())
            self.X_test = df
            self.length = len(self.keys)
            gradcam_path = os.path.join(data_dir, 'person_detections', "gradcams_test_person_pos_all.pkl")
            with open(gradcam_path, 'rb') as file:
                self.gradcam_dct = pickle.load(file)
            
        else:
            column_names = ['path', 'idx', 'body_bbox_x', 'body_bbox_y', 'body_bbox_w', 'body_bbox_h', 'eye_x', 'eye_y',
                            'gaze_x', 'gaze_y', 'bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max', 'inout', 'meta', 'ori_name',
                            'body_x1', 'body_y1', 'body_x2', 'body_y2']
            
            if ratio < 1.0:  
                df_unsup_ori = pd.read_csv(unsup_label, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")    
                df_unsup = df_unsup_ori.groupby(['path', 'eye_x'])
                df_sup_ori = pd.read_csv(sup_label, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")    
                df_sup = df_sup_ori.groupby(['path', 'eye_x'])
                self.sup_keys = set(df_sup.groups.keys())
                self.unsup_keys = set(df_unsup.groups.keys())
                self.df_sup = df_sup
                self.df_unsup = df_unsup
            
            
            df = pd.read_csv(csv_path, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
            df = df[df['inout'] != -1]  # only use "in" or "out "gaze. (-1 is invalid, 0 is out gaze)
            df = df[np.logical_and(np.less_equal(df['bbox_x_min'].values,df['bbox_x_max'].values), np.less_equal(df['bbox_y_min'].values, df['bbox_y_max'].values))]
            df.reset_index(inplace=True)
            
            self.df = df
            self.y_train = df[['bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max', 'eye_x', 'eye_y', 'gaze_x',
                            'gaze_y', 'inout', 'body_x1', 'body_y1', 'body_x2', 'body_y2']]
            self.X_train = df['path']
            self.length = len(df)
            gradcam_path = os.path.join(data_dir, 'person_detections', 'gradcams_train_person_pos_all.pkl')
            
            with open(gradcam_path, 'rb') as file:
                self.gradcam_dct = pickle.load(file)
            

        self.ratio = ratio
        self.data_dir = data_dir
        self.transform = transform
        self.test = test

        self.input_size = cfg.input_resolution
        self.output_size = cfg.output_resolution if not test else 64
        self.gradcam_output_size = cfg.gradcam_outsize
        self.gradcam_size = 30
        self.hm_sigma = 3
   
    def __getitem__(self, index):
        if self.test:
            g = self.X_test.get_group(self.keys[index])
            cont_gaze = []
            for i, row in g.iterrows():
                path = row['path']
                x_min = row['bbox_x_min']
                y_min = row['bbox_y_min']
                x_max = row['bbox_x_max']
                y_max = row['bbox_y_max']
                eye_x = row['eye_x']
                eye_y = row['eye_y']
                gaze_x = row['gaze_x']
                gaze_y = row['gaze_y']
                body_x_min = row['bbox_x_min']
                body_y_min = row['bbox_y_min']
                body_x_max = row['bbox_x_max']
                body_y_max = row['bbox_y_max']
                cont_gaze.append([gaze_x, gaze_y])  # all ground truth gaze are stacked up
            for j in range(len(cont_gaze), 20):
                cont_gaze.append([-1, -1])  # pad dummy gaze to match size for batch processing
            cont_gaze = torch.FloatTensor(cont_gaze)
            gaze_inside = True # always consider test samples as inside
        else:
            path = self.X_train.iloc[index]
            x_min, y_min, x_max, y_max, eye_x, eye_y, gaze_x, gaze_y, inout, body_x_min, body_y_min, body_x_max, body_y_max = self.y_train.iloc[index]
            
            if self.ratio==1.0:
                sup = 1
            else:
                if (path, eye_x) in self.sup_keys:  # use fully supervised label
                    sup = 1
                elif (path, eye_x) in self.unsup_keys:  # use pseudo label
                    sup = 0
                else:
                    print(f"{(path, eye_x)} not in either key!")
                    sup = -1
                
            gaze_inside = bool(inout)
        imgname = os.path.split(path)[-1]
        
        #if gaze_inside:
        if imgname not in self.gradcam_dct or eye_x not in self.gradcam_dct[imgname]:
            gradcam_this = torch.zeros((self.gradcam_output_size, self.gradcam_output_size))
            answer='none'
            valid=False
        else:
            gradcam_this = self.gradcam_dct[imgname][eye_x]['10th_layer']
            gradcam_this = torch.tensor(gradcam_this).float()
            answer = self.gradcam_dct[imgname][eye_x]['answer']
            valid = True
        #else:
            #answer='none'
            #valid = True
        crop, flip, jitter = False, False, False  
        #expand face bbox a bit
        k = 0.1
        x_min -= k * abs(x_max - x_min)
        y_min -= k * abs(y_max - y_min)
        x_max += k * abs(x_max - x_min)
        y_max += k * abs(y_max - y_min)

        img = Image.open(os.path.join(self.data_dir, path))
        img = img.convert('RGB')
        width, height = img.size
        x_min, y_min, x_max, y_max = map(float, [x_min, y_min, x_max, y_max])
        
        imsize = torch.IntTensor([width, height])
        if self.test:
            imsize = torch.IntTensor([width, height])
            
            
        elif not self.no_aug:
            ## data augmentation
            # Jitter (expansion-only) bounding box size
            if np.random.random_sample() <= 0.5:
                jitter = True
                k = np.random.random_sample() * 0.1  # changed from 0.2 to 0.1 here
                x_min -= k * abs(x_max - x_min)
                y_min -= k * abs(y_max - y_min)
                x_max += k * abs(x_max - x_min)
                y_max += k * abs(y_max - y_min)

            # Random Crop            
            if np.random.random_sample() <= 0.5 and sup==1:
                # modified: only crop for supervised cases as the gaze target is not known for the unsupervised images
                # Calculate the minimum valid range of the crop that doesn't exclude the face and the gaze target
                crop = True
                crop_x_min = np.min([gaze_x * width, x_min, x_max])
                crop_y_min = np.min([gaze_y * height, y_min, y_max])
                crop_x_max = np.max([gaze_x * width, x_min, x_max])
                crop_y_max = np.max([gaze_y * height, y_min, y_max])

                # Randomly select a random top left corner
                if crop_x_min >= 0:
                    crop_x_min = np.random.uniform(0, crop_x_min)
                if crop_y_min >= 0:
                    crop_y_min = np.random.uniform(0, crop_y_min)

                # Find the range of valid crop width and height starting from the (crop_x_min, crop_y_min)
                crop_width_min = crop_x_max - crop_x_min
                crop_height_min = crop_y_max - crop_y_min
                crop_width_max = width - crop_x_min
                crop_height_max = height - crop_y_min
                # Randomly select a width and a height
                crop_width = np.random.uniform(crop_width_min, crop_width_max)
                crop_height = np.random.uniform(crop_height_min, crop_height_max)
                
                # Record the crop's (x, y) offset
                offset_x, offset_y = crop_x_min, crop_y_min                
                # else:
                #     gaze_x = -1; gaze_y = -1
                if gaze_inside:
                    crop_xstart_ratio, crop_ystart_ratio = crop_x_min / width, crop_y_min / height
                    crop_ratio_w, crop_ratio_h = crop_width / width, crop_height / height
                    gradcam_crop_xstart, gradcam_crop_ystart = self.gradcam_size * crop_xstart_ratio, self.gradcam_size*crop_ystart_ratio
                    gradcam_crop_w, gradcam_crop_h = crop_ratio_w * self.gradcam_size, crop_ratio_h * self.gradcam_size
                    gradcam_crop_xend, gradcam_crop_yend = min(int(gradcam_crop_xstart+gradcam_crop_w), self.gradcam_size), min(int(gradcam_crop_ystart+gradcam_crop_h), self.gradcam_size)  # end coordinate (excluded)
                    gradcam_crop_xstart, gradcam_crop_ystart = int(gradcam_crop_xstart), int(gradcam_crop_ystart)
                    gradcam_crop_w = gradcam_crop_xend - gradcam_crop_xstart
                    gradcam_crop_h = gradcam_crop_yend - gradcam_crop_ystart
                    
                    if gradcam_crop_w>0 and gradcam_crop_h>0:
                        gradcam_new = torch.zeros((gradcam_crop_h, gradcam_crop_w))
                        copyfrom_x_start, copyfrom_y_start = max(0, gradcam_crop_xstart), max(0, gradcam_crop_ystart)
                        copyto_x_start, copyto_y_start = max(0, -gradcam_crop_xstart), max(0, -gradcam_crop_ystart)
                        gradcam_new[copyto_y_start:, copyto_x_start:] = gradcam_this[copyfrom_y_start:gradcam_crop_yend, copyfrom_x_start:gradcam_crop_xend]
                        gradcam_this = gradcam_new
                        
                    else:
                        crop=False # no crop then if gradcam width/height <= 0
                    
                if crop:
                    img = TF.crop(img, crop_y_min, crop_x_min, crop_height, crop_width)
                    # convert coordinates into the cropped frame
                    x_min, y_min, x_max, y_max = x_min - offset_x, y_min - offset_y, x_max - offset_x, y_max - offset_y
                    
                    body_x_min, body_y_min, body_x_max, body_y_max = body_x_min - offset_x, body_y_min - offset_y, body_x_max - offset_x, body_y_max - offset_y
                    # if gaze_inside:
                    gaze_x, gaze_y = (gaze_x * width - offset_x) / float(crop_width), \
                                    (gaze_y * height - offset_y) / float(crop_height)
                    width, height = crop_width, crop_height
                
            # Random flip
            if np.random.random_sample() <= 0.5:
                flip = True
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                x_max_2 = width - x_min
                x_min_2 = width - x_max
                body_x_max, body_x_min = width - body_x_min, width - body_x_max
                x_max = x_max_2
                x_min = x_min_2
                gaze_x = 1 - gaze_x
                #if gaze_inside:
                gradcam_this = torch.flip(gradcam_this, dims=[1])

            # Random color change
            if np.random.random_sample() <= 0.5:
                img = TF.adjust_brightness(img, brightness_factor=np.random.uniform(0.5, 1.5))
                img = TF.adjust_contrast(img, contrast_factor=np.random.uniform(0.5, 1.5))
                img = TF.adjust_saturation(img, saturation_factor=np.random.uniform(0, 1.5))
                
                
        head_channel = imutils.get_head_box_channel(x_min, y_min, x_max, y_max, width, height,
                                                    resolution=self.input_size, coordconv=False).unsqueeze(0)

        # Crop the face
        face = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
        body_coords = torch.tensor([body_x_min/width, body_y_min/height, body_x_max/width, body_y_max/height])

        if self.transform is not None:
            img = self.transform(img)
            face = self.transform(face)

        # generate the heat map used for deconv prediction
        gaze_heatmap = torch.zeros(self.output_size, self.output_size)  # set the size of the output
        gaze_heatmap_maxgradcam = torch.zeros(self.output_size, self.output_size)
        #gaze_heatmap_cp = torch.zeros(self.output_size, self.output_size)
        if self.test:  # aggregated heatmap
            gaze_coords = []
            num_valid = 0
            for gaze_x, gaze_y in cont_gaze:
                if gaze_x != -1:
                    num_valid += 1
                    #gaze_heatmap = imutils.draw_labelmap(gaze_heatmap, [gaze_x * self.output_size, gaze_y * self.output_size],
                                                         #3,
                                                         #type='Gaussian',
                                                         #maximum=True)
                    gaze_coords.append(torch.tensor([gaze_x, gaze_y]))
            #gaze_heatmap /= num_valid
            if num_valid>0:
                gaze_coords = torch.stack(gaze_coords).mean(dim=0)
                gaze_heatmap = imutils.draw_labelmap(gaze_heatmap, [gaze_coords[0].item() * self.output_size, gaze_coords[1].item() * self.output_size],
                                                         3,
                                                         type='Gaussian')
            else:
                gaze_coords = torch.tensor([-1.0, -1.0])
        else:
            if gaze_inside and sup==1:
                gaze_heatmap = imutils.draw_labelmap(gaze_heatmap, [gaze_x * self.output_size, gaze_y * self.output_size],
                                                    3,
                                                    type='Gaussian')
                gaze_coords = torch.tensor([gaze_x, gaze_y])
                    
            else:
                gaze_coords = torch.tensor([-1.0, -1.0])
                
        
        eye_pos = torch.tensor([eye_x,eye_y]) # only for test
        head_coords = torch.tensor([x_min/width, y_min/height, x_max/width, y_max/height])
        
        try:
            if gaze_inside:
                gradcam_this = gradcam_this.unsqueeze(0).unsqueeze(0)
                gradcam_resize = torch.nn.functional.interpolate(gradcam_this, size=self.gradcam_output_size, mode='bilinear', align_corners=False).squeeze()
                gradcam_resize = (gradcam_resize - gradcam_resize.min()) / (gradcam_resize.max() - gradcam_resize.min())
                if valid:
                    max_idx = torch.argmax(gradcam_this.flatten())
                    gc_h, gc_w = gradcam_this.size()[-2:]
                    max_x, max_y = max_idx % gc_w, torch.div(max_idx, gc_w, rounding_mode='floor')
                    max_x, max_y = max_x / gc_w, max_y / gc_h
                    gaze_heatmap_maxgradcam = imutils.draw_labelmap(gaze_heatmap_maxgradcam, [max_x.item() * self.output_size, max_y.item() * self.output_size],
                                                         3,
                                                         type='Gaussian')
                    
            else:
                gradcam_resize = torch.zeros((self.gradcam_output_size, self.gradcam_output_size))
                
            gradcam_resize = gradcam_resize.unsqueeze(0)
        except Exception:
            print(traceback.format_exc())
            pdb.set_trace()
            
        
        if self.test:
            return img, face, head_channel, gaze_heatmap, cont_gaze, imsize, head_coords, body_coords, gaze_coords, gradcam_resize, valid, path
        else:
            return img, face, head_channel, gaze_heatmap, gaze_inside, head_coords, body_coords, gaze_coords, gradcam_resize, gaze_heatmap_maxgradcam, sup, valid, path

    def __len__(self):
        return self.length


class GazeFollow_weakly_doubleaug(Dataset):
    def __init__(self, cfg, transform, test=False, ratio=0.5, use_sup_only=True, use_unsup_only=False, no_aug=False):
        data_dir = cfg.gazefollow_base_dir
        self.data_dir = data_dir
        self.no_aug = no_aug
        
        assert not (use_sup_only and use_unsup_only)
        if test:
            csv_path = os.path.join(data_dir, "test_annotations_release_persondet.txt") 
        elif (use_sup_only or use_unsup_only) and ratio!=1.0:
            if use_sup_only:
                csv_path = os.path.join(data_dir,  "weak_supervision", "train_annotations_weak_ratio{}_use.txt".format(ratio))
            else:
                csv_path = os.path.join(data_dir,  "weak_supervision", "train_annotations_weak_ratio{}_left.txt".format(ratio))
        else:
            csv_path = os.path.join(data_dir, "train_annotations_release_persondet.txt")
        
        if ratio<1.0:
            unsup_label, sup_label = os.path.join(data_dir,  "weak_supervision", "train_annotations_weak_ratio{}_left.txt".format(ratio)), os.path.join(data_dir,  "weak_supervision", "train_annotations_weak_ratio{}_use.txt".format(ratio))
            
        self.diff_pred_dir, self.baseline_pred_dir = os.path.join(data_dir, "diff_pred"), os.path.join(data_dir, "baseline_pred")
        
        if test:
            column_names = ['path', 'idx', 'body_bbox_x', 'body_bbox_y', 'body_bbox_w', 'body_bbox_h', 'eye_x', 'eye_y',
                            'gaze_x', 'gaze_y', 'bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max', 'meta', 'ori_name',
                            'body_x1', 'body_y1', 'body_x2', 'body_y2']
            df = pd.read_csv(csv_path, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
            df = df[['path', 'eye_x', 'eye_y', 'gaze_x', 'gaze_y', 'bbox_x_min', 'bbox_y_min', 'bbox_x_max',
                    'bbox_y_max']].groupby(['path', 'eye_x'])
            self.keys = list(df.groups.keys())
            self.X_test = df
            self.length = len(self.keys)
            gradcam_path = os.path.join(data_dir, 'person_detections', "gradcams_test_person_pos_all.pkl")
            with open(gradcam_path, 'rb') as file:
                self.gradcam_dct = pickle.load(file)
            
        else:
            column_names = ['path', 'idx', 'body_bbox_x', 'body_bbox_y', 'body_bbox_w', 'body_bbox_h', 'eye_x', 'eye_y',
                            'gaze_x', 'gaze_y', 'bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max', 'inout', 'meta', 'ori_name',
                            'body_x1', 'body_y1', 'body_x2', 'body_y2']
            if ratio < 1.0:  
                df_unsup_ori = pd.read_csv(unsup_label, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")    
                df_unsup = df_unsup_ori.groupby(['path', 'eye_x'])
                df_sup_ori = pd.read_csv(sup_label, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")    
                df_sup = df_sup_ori.groupby(['path', 'eye_x'])
                self.sup_keys = set(df_sup.groups.keys())
                self.unsup_keys = set(df_unsup.groups.keys())
                self.df_sup = df_sup
                self.df_unsup = df_unsup
            
            df = pd.concat([df_sup_ori, df_unsup_ori], ignore_index=True)
            df = df[df['inout'] != -1]  # only use "in" or "out "gaze. (-1 is invalid, 0 is out gaze)
            df = df[np.logical_and(np.less_equal(df['bbox_x_min'].values,df['bbox_x_max'].values), np.less_equal(df['bbox_y_min'].values, df['bbox_y_max'].values))]
            df.reset_index(inplace=True)
            
            self.y_train = df[['bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max', 'eye_x', 'eye_y', 'gaze_x',
                            'gaze_y', 'inout', 'body_x1', 'body_y1', 'body_x2', 'body_y2']]
            self.X_train = df['path']
            self.length = len(df)
            gradcam_path = os.path.join(data_dir, 'person_detections', 'gradcams_train_person_pos_all.pkl')
            with open(gradcam_path, 'rb') as file:
                self.gradcam_dct = pickle.load(file)

        self.ratio = ratio
        self.data_dir = data_dir
        self.transform = transform
        self.test = test

        self.input_size = cfg.input_resolution
        self.output_size = cfg.output_resolution
        self.gradcam_output_size = cfg.gradcam_outsize
        self.gradcam_size = 30
        self.hm_sigma = 3
   
    def __getitem__(self, index):
        if self.test:
            g = self.X_test.get_group(self.keys[index])
            cont_gaze = []
            for i, row in g.iterrows():
                path = row['path']
                x_min = row['bbox_x_min']
                y_min = row['bbox_y_min']
                x_max = row['bbox_x_max']
                y_max = row['bbox_y_max']
                eye_x = row['eye_x']
                eye_y = row['eye_y']
                gaze_x = row['gaze_x']
                gaze_y = row['gaze_y']
                body_x_min = row['bbox_x_min']
                body_y_min = row['bbox_y_min']
                body_x_max = row['bbox_x_max']
                body_y_max = row['bbox_y_max']
                cont_gaze.append([gaze_x, gaze_y])  # all ground truth gaze are stacked up
            for j in range(len(cont_gaze), 20):
                cont_gaze.append([-1, -1])  # pad dummy gaze to match size for batch processing
            cont_gaze = torch.FloatTensor(cont_gaze)
            gaze_inside = True # always consider test samples as inside
        else:
            path = self.X_train.iloc[index]
            x_min, y_min, x_max, y_max, eye_x, eye_y, gaze_x, gaze_y, inout, body_x_min, body_y_min, body_x_max, body_y_max = self.y_train.iloc[index]
            
            if self.ratio==1.0:
                sup = 1
            else:
                if (path, eye_x) in self.sup_keys:  # use fully supervised label
                    sup = 1
        
                elif (path, eye_x) in self.unsup_keys:  # use pseudo label
                    sup = 0
                else:
                    print(f"{(path, eye_x)} not in either key!")
                    sup = -1
                
            gaze_inside = bool(inout)
        imgname = os.path.split(path)[-1]
        
        if gaze_inside:
            if imgname not in self.gradcam_dct or eye_x not in self.gradcam_dct[imgname]:
                gradcam_this = torch.zeros((self.gradcam_output_size, self.gradcam_output_size))
            else:
                gradcam_this = self.gradcam_dct[imgname][eye_x]['10th_layer']
                gradcam_this = torch.tensor(gradcam_this).float()
        
        crop, flip, jitter = False, False, False  
        #expand face bbox a bit
        k = 0.1
        x_min -= k * abs(x_max - x_min)
        y_min -= k * abs(y_max - y_min)
        x_max += k * abs(x_max - x_min)
        y_max += k * abs(y_max - y_min)

        img_1 = Image.open(os.path.join(self.data_dir, path))
        img_1 = img_1.convert('RGB')
        width, height = img_1.size
        x_min, y_min, x_max, y_max = map(float, [x_min, y_min, x_max, y_max])
        
        #depth_path = os.path.join(self.depth_dir, path)
        #depth_path = depth_path[:-3]+'png'
        #depth_img = cv2.imread(depth_path, -1)

        #bits = 2
        #max_val = (2**(8*bits))-1
        #depth_img = depth_img / float(max_val)
        #depth_img = Image.fromarray(depth_img)
        
        x_min_1, y_min_1, x_max_1, y_max_1 = x_min, y_min, x_max, y_max
        x_min_2, y_min_2, x_max_2, y_max_2 = x_min, y_min, x_max, y_max 
        if self.test:
            imsize = torch.IntTensor([width, height])
        elif not self.no_aug:
            ## data augmentation
            # Jitter (expansion-only) bounding box size
            if np.random.random_sample() <= 0.5:
                jitter = True
                k = np.random.random_sample() * 0.1  # changed from 0.2 to 0.1 here
                x_min_1 = x_min - k * abs(x_max - x_min)
                y_min_1 = y_min - k * abs(y_max - y_min)
                x_max_1 = x_max + k * abs(x_max - x_min)
                y_max_1 = y_max + k * abs(y_max - y_min)
            
            if np.random.random_sample() <= 0.5:
                k = np.random.random_sample() * 0.1  # changed from 0.2 to 0.1 here    
                x_min_2 = x_min - k * abs(x_max - x_min)
                y_min_2 = y_min - k * abs(y_max - y_min)
                x_max_2 = x_max + k * abs(x_max - x_min)
                y_max_2 = y_max + k * abs(y_max - y_min)
                
            x_min_both, y_min_both, x_max_both, y_max_both = min(x_min_1, x_min_2), min(y_min_1, y_min_2), max(x_max_1, x_max_2), max(y_max_1, y_max_2)

            # Random Crop
            if np.random.random_sample() <= 0.5 and sup==1:
                # Calculate the minimum valid range of the crop that doesn't exclude the face and the gaze target
                crop = True
                crop_x_min = np.min([gaze_x * width, x_min_both, x_max_both])
                crop_y_min = np.min([gaze_y * height, y_min_both, y_max_both])
                crop_x_max = np.max([gaze_x * width, x_min_both, x_max_both])
                crop_y_max = np.max([gaze_y * height, y_min_both, y_max_both])

                # Randomly select a random top left corner
                if crop_x_min >= 0:
                    crop_x_min = np.random.uniform(0, crop_x_min)
                if crop_y_min >= 0:
                    crop_y_min = np.random.uniform(0, crop_y_min)

                # Find the range of valid crop width and height starting from the (crop_x_min, crop_y_min)
                crop_width_min = crop_x_max - crop_x_min
                crop_height_min = crop_y_max - crop_y_min
                crop_width_max = width - crop_x_min
                crop_height_max = height - crop_y_min
                # Randomly select a width and a height
                crop_width = np.random.uniform(crop_width_min, crop_width_max)
                crop_height = np.random.uniform(crop_height_min, crop_height_max)

                
                # Record the crop's (x, y) offset
                offset_x, offset_y = crop_x_min, crop_y_min
                
                # else:
                #     gaze_x = -1; gaze_y = -1
                if gaze_inside:
                    crop_xstart_ratio, crop_ystart_ratio = crop_x_min / width, crop_y_min / height
                    crop_ratio_w, crop_ratio_h = crop_width / width, crop_height / height
                    gradcam_crop_xstart, gradcam_crop_ystart = self.gradcam_size * crop_xstart_ratio, self.gradcam_size*crop_ystart_ratio
                    gradcam_crop_w, gradcam_crop_h = crop_ratio_w * self.gradcam_size, crop_ratio_h * self.gradcam_size
                    gradcam_crop_xend, gradcam_crop_yend = min(int(gradcam_crop_xstart+gradcam_crop_w), self.gradcam_size), min(int(gradcam_crop_ystart+gradcam_crop_h), self.gradcam_size)  # end coordinate (excluded)
                    gradcam_crop_xstart, gradcam_crop_ystart = int(gradcam_crop_xstart), int(gradcam_crop_ystart)
                    gradcam_crop_w = gradcam_crop_xend - gradcam_crop_xstart
                    gradcam_crop_h = gradcam_crop_yend - gradcam_crop_ystart
                    
                    if gradcam_crop_w>0 and gradcam_crop_h>0:
                        gradcam_new = torch.zeros((gradcam_crop_h, gradcam_crop_w))
                        copyfrom_x_start, copyfrom_y_start = max(0, gradcam_crop_xstart), max(0, gradcam_crop_ystart)
                        copyto_x_start, copyto_y_start = max(0, -gradcam_crop_xstart), max(0, -gradcam_crop_ystart)
                        gradcam_new[copyto_y_start:, copyto_x_start:] = gradcam_this[copyfrom_y_start:gradcam_crop_yend, copyfrom_x_start:gradcam_crop_xend]
                        gradcam_this = gradcam_new
                        
                    else:
                        crop=False # no crop then if gradcam width/height <= 0
                    
                if crop:
                    img_1 = TF.crop(img_1, crop_y_min, crop_x_min, crop_height, crop_width)
                    #depth_img = TF.crop(depth_img, crop_y_min, crop_x_min, crop_height, crop_width)
                    # convert coordinates into the cropped frame
                    x_min_1, y_min_1, x_max_1, y_max_1 = x_min_1 - offset_x, y_min_1 - offset_y, x_max_1 - offset_x, y_max_1 - offset_y
                    x_min_2, y_min_2, x_max_2, y_max_2 = x_min_2 - offset_x, y_min_2 - offset_y, x_max_2 - offset_x, y_max_2 - offset_y
                    
                    body_x_min, body_y_min, body_x_max, body_y_max = body_x_min - offset_x, body_y_min - offset_y, body_x_max - offset_x, body_y_max - offset_y
                    # if gaze_inside:
                    gaze_x, gaze_y = (gaze_x * width - offset_x) / float(crop_width), \
                                    (gaze_y * height - offset_y) / float(crop_height)
                    width, height = crop_width, crop_height
                
            # Random flip
            if np.random.random_sample() <= 0.5:
                flip = True
                img_1 = img_1.transpose(Image.FLIP_LEFT_RIGHT)
                #depth_img = depth_img.transpose(Image.FLIP_LEFT_RIGHT)
                x_max_1, x_min_1 = width - x_min_1, width - x_max_1
                x_max_2, x_min_2 = width - x_min_2, width - x_max_2
                body_x_max, body_x_min = width - body_x_min, width - body_x_max
                gaze_x = 1 - gaze_x
                if gaze_inside:
                    gradcam_this = torch.flip(gradcam_this, dims=[1])
                    
            img_2 = img_1.copy()
            # Random color change
            if np.random.random_sample() <= 0.5:
                img_1 = TF.adjust_brightness(img_1, brightness_factor=np.random.uniform(0.5, 1.5))
                img_1 = TF.adjust_contrast(img_1, contrast_factor=np.random.uniform(0.5, 1.5))
                img_1 = TF.adjust_saturation(img_1, saturation_factor=np.random.uniform(0, 1.5))
            
            if np.random.random_sample() <= 0.5:  
                img_2 = TF.adjust_brightness(img_2, brightness_factor=np.random.uniform(0.5, 1.5))
                img_2 = TF.adjust_contrast(img_2, contrast_factor=np.random.uniform(0.5, 1.5))
                img_2 = TF.adjust_saturation(img_2, saturation_factor=np.random.uniform(0, 1.5))


        head_channel_1 = imutils.get_head_box_channel(x_min_1, y_min_1, x_max_1, y_max_1, width, height,
                                                    resolution=self.input_size, coordconv=False).unsqueeze(0)
        head_channel_2 = imutils.get_head_box_channel(x_min_2, y_min_2, x_max_2, y_max_2, width, height,
                                                    resolution=self.input_size, coordconv=False).unsqueeze(0)

        # Crop the face
        face_1 = img_1.crop((int(x_min_1), int(y_min_1), int(x_max_1), int(y_max_1)))
        if not self.test:
            face_2 = img_2.crop((int(x_min_2), int(y_min_2), int(x_max_2), int(y_max_2)))
        body_coords = torch.tensor([body_x_min/width, body_y_min/height, body_x_max/width, body_y_max/height])

        if self.transform is not None:
            img_1 = self.transform(img_1)
            face_1 = self.transform(face_1)
            if not self.test:
                img_2 = self.transform(img_2)
                face_2 = self.transform(face_2)

        # generate the heat map used for deconv prediction
        gaze_heatmap = torch.zeros(self.output_size, self.output_size)  # set the size of the output
        
        if self.test:  # aggregated heatmap
            gaze_coords = []
            num_valid = 0
            for gaze_x, gaze_y in cont_gaze:
                if gaze_x != -1:
                    num_valid += 1
                    #gaze_heatmap = imutils.draw_labelmap(gaze_heatmap, [gaze_x * self.output_size, gaze_y * self.output_size],
                                                         #3,
                                                         #type='Gaussian',
                                                         #maximum=True)
                    gaze_coords.append(torch.tensor([gaze_x, gaze_y]))
            #gaze_heatmap /= num_valid
            if num_valid>0:
                gaze_coords = torch.stack(gaze_coords).mean(dim=0)
                gaze_heatmap = imutils.draw_labelmap(gaze_heatmap, [gaze_coords[0].item() * self.output_size, gaze_coords[1].item() * self.output_size],
                                                         3,
                                                         type='Gaussian')
            else:
                gaze_coords = torch.tensor([-1.0, -1.0])
        else:
            if gaze_inside and sup==1:
                gaze_heatmap = imutils.draw_labelmap(gaze_heatmap, [gaze_x * self.output_size, gaze_y * self.output_size],
                                                    3,
                                                    type='Gaussian')
                gaze_coords = torch.tensor([gaze_x, gaze_y])    
            else:
                gaze_coords = torch.tensor([-1.0, -1.0])

        
        eye_pos = torch.tensor([eye_x,eye_y]) # only for test
        head_coords = torch.tensor([x_min_1/width, y_min_1/height, x_max_1/width, y_max_1/height])
        #depth_img = depth_img.resize((self.input_size, self.input_size))
        #depth_img = torch.tensor(np.array(depth_img)).unsqueeze(0)
        
        try:
            if gaze_inside:
                gradcam_this = gradcam_this.unsqueeze(0).unsqueeze(0)
                gradcam_resize = torch.nn.functional.interpolate(gradcam_this, size=self.gradcam_output_size, mode='bilinear', align_corners=False).squeeze()
                gradcam_resize = (gradcam_resize - gradcam_resize.min()) / (gradcam_resize.max() - gradcam_resize.min())
                
            else:
                gradcam_resize = torch.zeros((self.gradcam_output_size, self.gradcam_output_size))
                
            gradcam_resize = gradcam_resize.unsqueeze(0)
        except Exception:
            print(traceback.format_exc())
            pdb.set_trace()
            
        
        if self.test:
            return img_1, face_1, head_channel_1, gaze_heatmap, cont_gaze, imsize, head_coords, body_coords, gaze_coords, gradcam_resize, path
        else:
            return img_1, face_1, head_channel_1, img_2, face_2, head_channel_2, gaze_heatmap, gaze_inside, head_coords, body_coords, gaze_coords, gradcam_resize, sup, path

    def __len__(self):
        return self.length




