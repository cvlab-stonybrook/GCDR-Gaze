import torch
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import pandas as pd
import pickle
import matplotlib as mpl
import pdb
import traceback
mpl.use('Agg')
from matplotlib import cm
import matplotlib.pyplot as plt
from skimage.transform import resize
import cv2
import glob
import csv
import pdb
from utils import imutils
from utils import myutils
from config import *
from datasets.select_clips_videoatt import clip_select


class VideoAttTarget_video_adaptation(Dataset):
    def __init__(self, args, transform,
                 test=False, show_name="", seq_len_limit=400, sup_only=False):
        
        super(VideoAttTarget_video_adaptation, self).__init__()
        self.base_dir = args.vat_base_dir
        self.data_dir = os.path.join(self.base_dir, 'images')
        annotation_dir = os.path.join(self.base_dir, 'annotations_body', 'test')
        #annotation_dir = os.path.join(self.base_dir, 'annt_predinout_test')
        self.split='train' if test is False else 'test'
        #shows = glob.glob(os.path.join(annotation_dir, '*'))
        #show = os.path.join(annotation_dir, 'test', show_name)
        show = os.path.join(annotation_dir, show_name)
        self.gradcams_dir = os.path.join(self.base_dir, 'gradcams_body', 'test', 'gradcams')
        self.clips_selected = clip_select[show_name]
        self.depth_dir = args.vat_depth_dir
        print(show_name) 
        train_sequence_paths = []
        for clip in self.clips_selected:
            if show_name=='MLB_interview':
                train_sequence_paths = [os.path.join(show, clip, 's00.txt')]
            else:
                sequence_annotations = glob.glob(os.path.join(show, clip, '*.txt'))  # each annotation file is for one person
                train_sequence_paths.extend(sequence_annotations)
        
        test_sequence_paths = glob.glob(os.path.join(show, '*', '*.txt'))  # each annotation file is for one person
        self.train_sequence_paths = train_sequence_paths    
        self.test_sequence_paths = test_sequence_paths    
        if test or not sup_only:
            self.all_sequence_paths = test_sequence_paths # semi-supervised adaptation uses the whole video sequence
        else:
            self.all_sequence_paths = train_sequence_paths
        self.transform = transform
        self.input_size = args.input_resolution
        self.output_size = args.output_resolution 
        self.test = test
        self.length = len(self.all_sequence_paths)
        self.seq_len_limit = seq_len_limit    
        self.show_name = show_name
        self.gradcam_size = 30
        #self.max_supervised = args.max_supervised

    def __getitem__(self, index):
        sequence_path = self.all_sequence_paths[index]
        sup=True if sequence_path in self.train_sequence_paths else False
        seq_indices = []
         
        df = pd.read_csv(sequence_path, header=None, index_col=False,
                         names=['path', 'xmin', 'ymin', 'xmax', 'ymax', 'gazex', 'gazey', 'body_x1', 'body_y1', 'body_x2', 'body_y2', 'pred_inout'])
        person_name = os.path.basename(sequence_path).split('.')[0]
        show_name = sequence_path.split('/')[-3]
        clip = sequence_path.split('/')[-2]
        seq_len = len(df.index)
        # moving-avg smoothing
        window_size = 11 # should be odd number
        df['xmin'] = myutils.smooth_by_conv(window_size, df, 'xmin')
        df['ymin'] = myutils.smooth_by_conv(window_size, df, 'ymin')
        df['xmax'] = myutils.smooth_by_conv(window_size, df, 'xmax')
        df['ymax'] = myutils.smooth_by_conv(window_size, df, 'ymax')
        gradcam_path = os.path.join(self.gradcams_dir, show_name, clip, f"{person_name}.pkl")
        with open(gradcam_path, 'rb') as file:
            gradcam_dct = pickle.load(file)
        
        if not self.test:
            # cond for data augmentation
            cond_jitter = np.random.random_sample()
            cond_flip = np.random.random_sample()
            cond_color = np.random.random_sample()
            if cond_color < 0.5:
                n1 = np.random.uniform(0.5, 1.5)
                n2 = np.random.uniform(0.5, 1.5)
                n3 = np.random.uniform(0.5, 1.5)
            cond_crop = np.random.random_sample()

            # if longer than seq_len_limit, cut it down to the limit with the init index randomly sampled
            if seq_len > self.seq_len_limit:
                sampled_ind = np.random.randint(0, seq_len - self.seq_len_limit)
                seq_len = self.seq_len_limit
            else:
                sampled_ind = 0

            if cond_crop < 0.5 and sup==True:
                sliced_x_min = df['xmin'].iloc[sampled_ind:sampled_ind+seq_len]
                sliced_x_max = df['xmax'].iloc[sampled_ind:sampled_ind+seq_len]
                sliced_y_min = df['ymin'].iloc[sampled_ind:sampled_ind+seq_len]
                sliced_y_max = df['ymax'].iloc[sampled_ind:sampled_ind+seq_len]

                sliced_gaze_x = df['gazex'].iloc[sampled_ind:sampled_ind+seq_len]
                sliced_gaze_y = df['gazey'].iloc[sampled_ind:sampled_ind+seq_len]

                check_sum = sliced_gaze_x.sum() + sliced_gaze_y.sum()
                all_outside = check_sum == -2*seq_len

                # Calculate the minimum valid range of the crop that doesn't exclude the face and the gaze target
                if all_outside:
                    crop_x_min = np.min([sliced_x_min.min(), sliced_x_max.min()])
                    crop_y_min = np.min([sliced_y_min.min(), sliced_y_max.min()])
                    crop_x_max = np.max([sliced_x_min.max(), sliced_x_max.max()])
                    crop_y_max = np.max([sliced_y_min.max(), sliced_y_max.max()])
                else:
                    crop_x_min = np.min([sliced_gaze_x.min(), sliced_x_min.min(), sliced_x_max.min()])
                    crop_y_min = np.min([sliced_gaze_y.min(), sliced_y_min.min(), sliced_y_max.min()])
                    crop_x_max = np.max([sliced_gaze_x.max(), sliced_x_min.max(), sliced_x_max.max()])
                    crop_y_max = np.max([sliced_gaze_y.max(), sliced_y_min.max(), sliced_y_max.max()])

                # Randomly select a random top left corner
                if crop_x_min >= 0:
                    crop_x_min = np.random.uniform(0, crop_x_min)
                if crop_y_min >= 0:
                    crop_y_min = np.random.uniform(0, crop_y_min)

                # Get image size
                path = os.path.join(self.data_dir, show_name, clip, df['path'].iloc[0])
                img = Image.open(path)
                img = img.convert('RGB')
                width, height = img.size

                # Find the range of valid crop width and height starting from the (crop_x_min, crop_y_min)
                crop_width_min = crop_x_max - crop_x_min
                crop_height_min = crop_y_max - crop_y_min
                crop_width_max = width - crop_x_min
                crop_height_max = height - crop_y_min
                # Randomly select a width and a height
                crop_width = np.random.uniform(crop_width_min, crop_width_max)
                crop_height = np.random.uniform(crop_height_min, crop_height_max)
        else:
            sampled_ind = 0

        
        faces, images, head_channels, heatmaps, gradcams, paths, gazes, imsizes, head_coords, gaze_inouts, gradcams_valid = [], [], [], [], [], [], [], [], [], [], []
        index_tracker = -1
        sup_list = []   # whether each frame is supervised or not
        depth_images = []
        pred_inout_seq = []
        heatmaps_maxgradcam = []
        num_inside_samples = 0

        for i, row in df.iterrows():
            index_tracker = index_tracker+1
            if not self.test:
                if index_tracker < sampled_ind or index_tracker >= (sampled_ind + self.seq_len_limit):
                    continue
            seq_indices.append(index)
            face_x1 = row['xmin']  # note: Already in image coordinates
            face_y1 = row['ymin']  # note: Already in image coordinates
            face_x2 = row['xmax']  # note: Already in image coordinates
            face_y2 = row['ymax']  # note: Already in image coordinates
            gaze_x = row['gazex']  # note: Already in image coordinates
            gaze_y = row['gazey']  # note: Already in image coordinates
            pred_inout = row['pred_inout']
            pred_inout_seq.append(torch.tensor(pred_inout))
            
            impath = os.path.join(self.data_dir, show_name, clip, row['path'])
            img = Image.open(impath)
            img = img.convert('RGB')
            imgname = os.path.splitext(os.path.basename(impath))[0]
            paths.append(impath)

            depth_path = os.path.join(self.depth_dir, show_name, clip, row['path'])
            depth_path = depth_path[:-3]+'png'
            depth_img = cv2.imread(depth_path, -1)
            bits = 2
            max_val = (2**(8*bits))-1
            depth_img = depth_img / float(max_val)
            depth_img = Image.fromarray(depth_img)

            
            width, height = img.size
            imsize = torch.FloatTensor([width, height])
            imsizes.append(imsize)

            face_x1, face_y1, face_x2, face_y2 = map(float, [face_x1, face_y1, face_x2, face_y2])
            gaze_x, gaze_y = map(float, [gaze_x, gaze_y])
            if gaze_x == -1 and gaze_y == -1:
                gaze_inside = False
            else:
                if gaze_x < 0: # move gaze point that was sliglty outside the image back in
                    gaze_x = 0
                if gaze_y < 0:
                    gaze_y = 0
                gaze_inside = True
            
            if gaze_inside:
                if imgname not in gradcam_dct:
                    gradcam_this = torch.zeros((self.output_size, self.output_size))
                    answer='none'
                    valid=False
                else:
                    gradcam_this = gradcam_dct[imgname]['10th_layer']
                    gradcam_this = torch.tensor(gradcam_this).float()
                    answer = gradcam_dct[imgname]['answer']
                    valid = True
            else:
                answer='none'
                valid = True
                        
            num_inside_samples+=1

            if not self.test:
                ## data augmentation
                # Jitter (expansion-only) bounding box size.
                if cond_jitter < 0.5:
                    k = cond_jitter * 0.1
                    face_x1 -= k * abs(face_x2 - face_x1)
                    face_y1 -= k * abs(face_y2 - face_y1)
                    face_x2 += k * abs(face_x2 - face_x1)
                    face_y2 += k * abs(face_y2 - face_y1)
                    face_x1 = np.clip(face_x1, 0, width)
                    face_x2 = np.clip(face_x2, 0, width)
                    face_y1 = np.clip(face_y1, 0, height)
                    face_y2 = np.clip(face_y2, 0, height)

                # Random Crop
                if cond_crop < 0.5 and sup==True:
                    # Crop it
                    img = TF.crop(img, crop_y_min, crop_x_min, crop_height, crop_width)
                    depth_img = TF.crop(depth_img, crop_y_min, crop_x_min, crop_height, crop_width)
                    # Record the crop's (x, y) offset
                    offset_x, offset_y = crop_x_min, crop_y_min

                    # convert coordinates into the cropped frame
                    face_x1, face_y1, face_x2, face_y2 = face_x1 - offset_x, face_y1 - offset_y, face_x2 - offset_x, face_y2 - offset_y
                    if gaze_inside:
                        gaze_x, gaze_y = (gaze_x- offset_x), \
                                         (gaze_y - offset_y)
                    else:
                        gaze_x = -1; gaze_y = -1
                        
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
                    width, height = crop_width, crop_height
                    
                # Flip?
                if cond_flip < 0.5:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    depth_img = depth_img.transpose(Image.FLIP_LEFT_RIGHT)
                    x_max_2 = width - face_x1
                    x_min_2 = width - face_x2
                    face_x2 = x_max_2
                    face_x1 = x_min_2
                    if gaze_x != -1 and gaze_y != -1:
                        gaze_x = width - gaze_x
                    if gaze_inside:
                        gradcam_this = torch.flip(gradcam_this, dims=[1])

                # Random color change
                if cond_color < 0.5:
                    img = TF.adjust_brightness(img, brightness_factor=n1)
                    img = TF.adjust_contrast(img, contrast_factor=n2)
                    img = TF.adjust_saturation(img, saturation_factor=n3)

            # Face crop
            face = img.copy().crop((int(face_x1), int(face_y1), int(face_x2), int(face_y2)))
            
            # Head channel image
            head_channel = imutils.get_head_box_channel(face_x1, face_y1, face_x2, face_y2, width, height,
                                                        resolution=self.input_size, coordconv=False).unsqueeze(0)
            if self.transform is not None:
                img = self.transform(img)
                face = self.transform(face)

            # Deconv output
            if gaze_inside:
                gaze_x /= float(width) # fractional gaze
                gaze_y /= float(height)
                gaze_heatmap = torch.zeros(self.output_size, self.output_size)  # set the size of the output
                gaze_map = imutils.draw_labelmap(gaze_heatmap, [gaze_x * self.output_size, gaze_y * self.output_size],
                                                 3,
                                                 type='Gaussian')
                gazes.append(torch.FloatTensor([gaze_x, gaze_y]))
            else:
                gaze_map = torch.zeros(self.output_size, self.output_size)
                gazes.append(torch.FloatTensor([-1, -1]))
            
            
            gaze_heatmap_maxgradcam = torch.zeros(self.output_size, self.output_size)
            try:
                if gaze_inside:
                    gradcam_this = gradcam_this.unsqueeze(0).unsqueeze(0)
                    gradcam_resize = torch.nn.functional.interpolate(gradcam_this, size=self.output_size, mode='bilinear', align_corners=False).squeeze()
                    gradcam_resize = (gradcam_resize - gradcam_resize.min()) / (gradcam_resize.max() - gradcam_resize.min())
                    if valid:
                        max_idx = torch.argmax(gradcam_this.flatten())
                        gc_h, gc_w = gradcam_this.size()[-2:]
                        max_x, max_y = max_idx.item() % gc_w, max_idx.item() // gc_w
                        max_x, max_y = max_x / gc_w, max_y / gc_h
                        gaze_heatmap_maxgradcam = imutils.draw_labelmap(gaze_heatmap_maxgradcam, [max_x* self.output_size, max_y * self.output_size],
                                                            3,
                                                            type='Gaussian')
                    
                else:
                    gradcam_resize = torch.zeros((self.output_size, self.output_size))
                    
                gradcam_resize = gradcam_resize.unsqueeze(0)
            except Exception:
                print(traceback.format_exc())
                pdb.set_trace()    
            
            
            depth_img = depth_img.resize((self.input_size, self.input_size))
            depth_images.append(torch.tensor(np.array(depth_img)).unsqueeze(0))
            faces.append(face)
            images.append(img)
            head_channels.append(head_channel)
            heatmaps.append(gaze_map)
            gaze_inouts.append(torch.FloatTensor([int(gaze_inside)]))
            gradcams.append(gradcam_resize)
            gradcams_valid.append(torch.tensor(valid))
            head_coords.append(torch.FloatTensor([face_x1, face_y1, face_x2, face_y2]))
            heatmaps_maxgradcam.append(gaze_heatmap_maxgradcam)
            
                    
        seq_indices = torch.tensor(seq_indices).view(-1,1)
        faces = torch.stack(faces)
        images = torch.stack(images)
        head_channels = torch.stack(head_channels)
        heatmaps = torch.stack(heatmaps)
        gazes = torch.stack(gazes)
        gaze_inouts = torch.stack(gaze_inouts)
        depth_images = torch.stack(depth_images)
        gradcams = torch.stack(gradcams)
        gradcams_valid = torch.stack(gradcams_valid)
        imsizes = torch.stack(imsizes)
        head_coords = torch.stack(head_coords)
        pred_inout_seq = torch.stack(pred_inout_seq).view(-1, 1)
        heatmaps_maxgradcam = torch.stack(heatmaps_maxgradcam)
        
        # imsizes = torch.stack(imsizes)
        # print(faces.shape, images.shape, head_channels.shape, heatmaps.shape) 
        if self.test:
            return images, faces, head_channels, depth_images, heatmaps, gazes, head_coords, gaze_inouts, imsizes, gradcams, heatmaps_maxgradcam, gradcams_valid,sup
        else:
            return images, faces, head_channels, depth_images, heatmaps, gazes, head_coords, gaze_inouts, gradcams, heatmaps_maxgradcam, gradcams_valid, pred_inout_seq, sup

    def __len__(self):
        return self.length


def compose_images_in_videoannt(all_sequence_paths):
    
    all_data_info = []
    
    for annt_file in all_sequence_paths:
        df = pd.read_csv(annt_file, header=None, index_col=False,
                         names=['path', 'xmin', 'ymin', 'xmax', 'ymax', 'gazex', 'gazey', 'body_x1', 'body_y1', 'body_x2', 'body_y2'])
        all_data_info.append()



class VideoAttTarget_video_weakly(Dataset):
    def __init__(self, args, transform, input_size=input_resolution, output_size=output_resolution,
                 test=False, imshow=False, seq_len_limit=400, sup_only=False, unsup_only=False, no_aug=False):
        
        assert not (sup_only and unsup_only)
        self.test = test
        self.split='train' if test is False else 'test'
        self.base_dir = args.vat_base_dir
        data_dir = os.path.join(self.base_dir, 'images')
        annotation_dir = os.path.join(self.base_dir, 'annotations_body', self.split)
        shows = glob.glob(os.path.join(annotation_dir, '*'))
        self.all_sequence_paths = []
        self.supervised_videos = ['Band of Brothers', 'Modern Family', 'Coveted', 'Big Bang Theory']
        self.sup_only = sup_only
        sup_shows = []
        if not self.test:
            sup_shows = [os.path.join(annotation_dir, show) for show in self.supervised_videos]
            if sup_only:
                shows = sup_shows
            elif unsup_only:
                shows = [show for show in shows if show not in sup_shows]
        
        sup_labels_person = []   # each annt is a sequence for one person in a video
        for s in shows:
            sequence_annotations = glob.glob(os.path.join(s, '*', '*.txt'))  # each annotation file is for one person
            self.all_sequence_paths.extend(sequence_annotations)
            if s in sup_shows:
                sup_labels_person.extend([1]*len(sequence_annotations))
            else:
                sup_labels_person.extend([0]*len(sequence_annotations))
        self.sup_labels_person = sup_labels_person
            
        self.data_dir = data_dir
        self.depth_dir = args.vat_depth_dir
        self.transform = transform
        self.input_size = args.input_resolution
        self.output_size = args.output_resolution 
        self.test = test
        self.no_aug = no_aug
        self.length = len(self.all_sequence_paths)
        self.seq_len_limit = seq_len_limit    
        self.gradcam_size = 30
        self.gradcams_dir = os.path.join(self.base_dir, 'gradcams_body_new', self.split, 'gradcams')

    def __getitem__(self, index):
        sequence_path = self.all_sequence_paths[index]
        sup=True if self.sup_labels_person[index] or self.test else False
        seq_indices = []
         
        df = pd.read_csv(sequence_path, header=None, index_col=False,
                         names=['path', 'xmin', 'ymin', 'xmax', 'ymax', 'gazex', 'gazey', 'body_x1', 'body_y1', 'body_x2', 'body_y2', 'pred_inout'])
        person_name = os.path.basename(sequence_path).split('.')[0]
        show_name = sequence_path.split('/')[-3]
        clip = sequence_path.split('/')[-2]
        seq_len = len(df.index)
        # moving-avg smoothing
        window_size = 11 # should be odd number
        df['xmin'] = myutils.smooth_by_conv(window_size, df, 'xmin')
        df['ymin'] = myutils.smooth_by_conv(window_size, df, 'ymin')
        df['xmax'] = myutils.smooth_by_conv(window_size, df, 'xmax')
        df['ymax'] = myutils.smooth_by_conv(window_size, df, 'ymax')
        gradcam_path = os.path.join(self.gradcams_dir, show_name, clip, f"{person_name}.pkl")
        gradcam_exist = True if os.path.exists(gradcam_path) else False
        
        if gradcam_exist:
            with open(gradcam_path, 'rb') as file:
                gradcam_dct = pickle.load(file)
        
        if not self.test and not self.no_aug:
            # cond for data augmentation
            cond_jitter = np.random.random_sample()
            cond_flip = np.random.random_sample()
            cond_color = np.random.random_sample()
            if cond_color < 0.5:
                n1 = np.random.uniform(0.5, 1.5)
                n2 = np.random.uniform(0.5, 1.5)
                n3 = np.random.uniform(0.5, 1.5)
            cond_crop = np.random.random_sample()

            # if longer than seq_len_limit, cut it down to the limit with the init index randomly sampled
            if seq_len > self.seq_len_limit:
                sampled_ind = np.random.randint(0, seq_len - self.seq_len_limit)
                seq_len = self.seq_len_limit
            else:
                sampled_ind = 0

            if cond_crop < 0.5 and sup:
                sliced_x_min = df['xmin'].iloc[sampled_ind:sampled_ind+seq_len]
                sliced_x_max = df['xmax'].iloc[sampled_ind:sampled_ind+seq_len]
                sliced_y_min = df['ymin'].iloc[sampled_ind:sampled_ind+seq_len]
                sliced_y_max = df['ymax'].iloc[sampled_ind:sampled_ind+seq_len]

                sliced_gaze_x = df['gazex'].iloc[sampled_ind:sampled_ind+seq_len]
                sliced_gaze_y = df['gazey'].iloc[sampled_ind:sampled_ind+seq_len]

                check_sum = sliced_gaze_x.sum() + sliced_gaze_y.sum()
                all_outside = check_sum == -2*seq_len

                # Calculate the minimum valid range of the crop that doesn't exclude the face and the gaze target
                if all_outside:
                    crop_x_min = np.min([sliced_x_min.min(), sliced_x_max.min()])
                    crop_y_min = np.min([sliced_y_min.min(), sliced_y_max.min()])
                    crop_x_max = np.max([sliced_x_min.max(), sliced_x_max.max()])
                    crop_y_max = np.max([sliced_y_min.max(), sliced_y_max.max()])
                else:
                    crop_x_min = np.min([sliced_gaze_x.min(), sliced_x_min.min(), sliced_x_max.min()])
                    crop_y_min = np.min([sliced_gaze_y.min(), sliced_y_min.min(), sliced_y_max.min()])
                    crop_x_max = np.max([sliced_gaze_x.max(), sliced_x_min.max(), sliced_x_max.max()])
                    crop_y_max = np.max([sliced_gaze_y.max(), sliced_y_min.max(), sliced_y_max.max()])

                # Randomly select a random top left corner
                if crop_x_min >= 0:
                    crop_x_min = np.random.uniform(0, crop_x_min)
                if crop_y_min >= 0:
                    crop_y_min = np.random.uniform(0, crop_y_min)

                # Get image size
                path = os.path.join(self.data_dir, show_name, clip, df['path'].iloc[0])
                img = Image.open(path)
                img = img.convert('RGB')
                width, height = img.size

                # Find the range of valid crop width and height starting from the (crop_x_min, crop_y_min)
                crop_width_min = crop_x_max - crop_x_min
                crop_height_min = crop_y_max - crop_y_min
                crop_width_max = width - crop_x_min
                crop_height_max = height - crop_y_min
                # Randomly select a width and a height
                crop_width = np.random.uniform(crop_width_min, crop_width_max)
                crop_height = np.random.uniform(crop_height_min, crop_height_max)
        else:
            sampled_ind = 0

        
        faces, images, head_channels, heatmaps, gradcams, paths, gazes, imsizes, head_coords, gaze_inouts, gradcams_valid = [], [], [], [], [], [], [], [], [], [], []
        index_tracker = -1
        sup_list = []   # whether each frame is supervised or not
        depth_images = []
        pred_inout_seq = []
        heatmaps_maxgradcam = []
        num_inside_samples = 0

        for i, row in df.iterrows():
            index_tracker = index_tracker+1
            if not self.test:
                if index_tracker < sampled_ind or index_tracker >= (sampled_ind + self.seq_len_limit):
                    continue
            seq_indices.append(index)
            face_x1 = row['xmin']  # note: Already in image coordinates
            face_y1 = row['ymin']  # note: Already in image coordinates
            face_x2 = row['xmax']  # note: Already in image coordinates
            face_y2 = row['ymax']  # note: Already in image coordinates
            gaze_x = row['gazex']  # note: Already in image coordinates
            gaze_y = row['gazey']  # note: Already in image coordinates
            pred_inout = row['pred_inout']
            pred_inout_seq.append(torch.tensor(pred_inout))
            
            impath = os.path.join(self.data_dir, show_name, clip, row['path'])
            img = Image.open(impath)
            img = img.convert('RGB')
            imgname = os.path.splitext(os.path.basename(impath))[0]
            paths.append(impath)

            depth_path = os.path.join(self.depth_dir, show_name, clip, row['path'])
            depth_path = depth_path[:-3]+'png'
            depth_img = cv2.imread(depth_path, -1)
            bits = 2
            max_val = (2**(8*bits))-1
            depth_img = depth_img / float(max_val)
            depth_img = Image.fromarray(depth_img)
            
            width, height = img.size
            imsize = torch.FloatTensor([width, height])
            imsizes.append(imsize)

            face_x1, face_y1, face_x2, face_y2 = map(float, [face_x1, face_y1, face_x2, face_y2])
            gaze_x, gaze_y = map(float, [gaze_x, gaze_y])
            if gaze_x == -1 and gaze_y == -1:
                gaze_inside = False
            else:
                if gaze_x < 0: # move gaze point that was sliglty outside the image back in
                    gaze_x = 0
                if gaze_y < 0:
                    gaze_y = 0
                gaze_inside = True
            
            if gaze_inside:
                if imgname not in gradcam_dct:
                    gradcam_this = torch.zeros((self.output_size, self.output_size))
                    answer='none'
                    valid=False
                else:
                    gradcam_this = gradcam_dct[imgname]['10th_layer']
                    gradcam_this = torch.tensor(gradcam_this).float()
                    answer = gradcam_dct[imgname]['answer']
                    valid = True
            else:
                answer='none'
                valid = False
                        
            num_inside_samples+=1

            if not self.test and not self.no_aug:
                ## data augmentation
                # Jitter (expansion-only) bounding box size.
                if cond_jitter < 0.5:
                    k = cond_jitter * 0.1
                    face_x1 -= k * abs(face_x2 - face_x1)
                    face_y1 -= k * abs(face_y2 - face_y1)
                    face_x2 += k * abs(face_x2 - face_x1)
                    face_y2 += k * abs(face_y2 - face_y1)
                    face_x1 = np.clip(face_x1, 0, width)
                    face_x2 = np.clip(face_x2, 0, width)
                    face_y1 = np.clip(face_y1, 0, height)
                    face_y2 = np.clip(face_y2, 0, height)

                # Random Crop
                if cond_crop < 0.5 and sup:
                    # Crop it
                    img = TF.crop(img, crop_y_min, crop_x_min, crop_height, crop_width)
                    depth_img = TF.crop(depth_img, crop_y_min, crop_x_min, crop_height, crop_width)
                    # Record the crop's (x, y) offset
                    offset_x, offset_y = crop_x_min, crop_y_min

                    # convert coordinates into the cropped frame
                    face_x1, face_y1, face_x2, face_y2 = face_x1 - offset_x, face_y1 - offset_y, face_x2 - offset_x, face_y2 - offset_y
                    if gaze_inside:
                        gaze_x, gaze_y = (gaze_x- offset_x), \
                                         (gaze_y - offset_y)
                    else:
                        gaze_x = -1; gaze_y = -1
                        
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
                    width, height = crop_width, crop_height
                    
                # Flip?
                if cond_flip < 0.5:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    depth_img = depth_img.transpose(Image.FLIP_LEFT_RIGHT)
                    x_max_2 = width - face_x1
                    x_min_2 = width - face_x2
                    face_x2 = x_max_2
                    face_x1 = x_min_2
                    if gaze_x != -1 and gaze_y != -1:
                        gaze_x = width - gaze_x
                    if gaze_inside:
                        gradcam_this = torch.flip(gradcam_this, dims=[1])

                # Random color change
                if cond_color < 0.5:
                    img = TF.adjust_brightness(img, brightness_factor=n1)
                    img = TF.adjust_contrast(img, contrast_factor=n2)
                    img = TF.adjust_saturation(img, saturation_factor=n3)

            # Face crop
            face = img.copy().crop((int(face_x1), int(face_y1), int(face_x2), int(face_y2)))
            
            # Head channel image
            head_channel = imutils.get_head_box_channel(face_x1, face_y1, face_x2, face_y2, width, height,
                                                        resolution=self.input_size, coordconv=False).unsqueeze(0)
            if self.transform is not None:
                img = self.transform(img)
                face = self.transform(face)

            # Deconv output
            if gaze_inside:
                gaze_x /= float(width) # fractional gaze
                gaze_y /= float(height)
                gaze_heatmap = torch.zeros(self.output_size, self.output_size)  # set the size of the output
                gaze_map = imutils.draw_labelmap(gaze_heatmap, [gaze_x * self.output_size, gaze_y * self.output_size],
                                                 3,
                                                 type='Gaussian')
                gazes.append(torch.FloatTensor([gaze_x, gaze_y]))
            else:
                gaze_map = torch.zeros(self.output_size, self.output_size)
                gazes.append(torch.FloatTensor([-1, -1]))
            
            
            gaze_heatmap_maxgradcam = torch.zeros(self.output_size, self.output_size)
            try:
                if gaze_inside and valid:
                    gradcam_this = gradcam_this.unsqueeze(0).unsqueeze(0)
                    gradcam_resize = torch.nn.functional.interpolate(gradcam_this, size=self.output_size, mode='bilinear', align_corners=False).squeeze()
                    gradcam_resize = (gradcam_resize - gradcam_resize.min()) / (gradcam_resize.max() - gradcam_resize.min())
                   
                    max_idx = torch.argmax(gradcam_this.flatten())
                    gc_h, gc_w = gradcam_this.size()[-2:]
                    max_x, max_y = max_idx.item() % gc_w, max_idx.item() // gc_w
                    max_x, max_y = max_x / gc_w, max_y / gc_h
                    gaze_heatmap_maxgradcam = imutils.draw_labelmap(gaze_heatmap_maxgradcam, [max_x* self.output_size, max_y * self.output_size],
                                                        3,
                                                        type='Gaussian')
                    
                else:
                    gradcam_resize = torch.zeros((self.output_size, self.output_size))
                    
                gradcam_resize = gradcam_resize.unsqueeze(0)
            except Exception:
                print(traceback.format_exc())
                pdb.set_trace()    
            
            
            depth_img = depth_img.resize((self.input_size, self.input_size))
            depth_images.append(torch.tensor(np.array(depth_img)).unsqueeze(0))
            faces.append(face)
            images.append(img)
            head_channels.append(head_channel)
            heatmaps.append(gaze_map)
            gaze_inouts.append(torch.FloatTensor([int(gaze_inside)]))
            gradcams.append(gradcam_resize)
            gradcams_valid.append(torch.tensor(valid))
            head_coords.append(torch.FloatTensor([face_x1, face_y1, face_x2, face_y2]))
            heatmaps_maxgradcam.append(gaze_heatmap_maxgradcam)
            
                    
        seq_indices = torch.tensor(seq_indices).view(-1,1)
        faces = torch.stack(faces)
        images = torch.stack(images)
        head_channels = torch.stack(head_channels)
        heatmaps = torch.stack(heatmaps)
        gazes = torch.stack(gazes)
        gaze_inouts = torch.stack(gaze_inouts)
        depth_images = torch.stack(depth_images)
        gradcams = torch.stack(gradcams)
        gradcams_valid = torch.stack(gradcams_valid)
        imsizes = torch.stack(imsizes)
        head_coords = torch.stack(head_coords)
        pred_inout_seq = torch.stack(pred_inout_seq).view(-1, 1)
        heatmaps_maxgradcam = torch.stack(heatmaps_maxgradcam)
        
        # imsizes = torch.stack(imsizes)
        # print(faces.shape, images.shape, head_channels.shape, heatmaps.shape) 
        if self.test:
            return images, faces, head_channels, depth_images, heatmaps, gazes, head_coords, gaze_inouts, imsizes, gradcams, heatmaps_maxgradcam, gradcams_valid,sup
        else:
            return images, faces, head_channels, depth_images, heatmaps, gazes, head_coords, gaze_inouts, gradcams, heatmaps_maxgradcam, gradcams_valid, sup

    def __len__(self):
        return self.length




def get_Videoatt_Seq(seq_len = 5):
    data_dir = '/nfs/bigbrain/qiaomu/datasets/gaze/videoattentiontarget'
    anno_dir = os.path.join(data_dir, 'annotations')
    anno_seq_train, anno_seq_test = [],[]
    for mode in ['train', 'test']:
        if mode=='train':
            anno_seq = anno_seq_train
        else:
            anno_seq = anno_seq_test
        num_seq, num_frame = 0,0
        all_annot = glob.glob(os.path.join(anno_dir, mode, '*', '*', '*.txt'))   
        for anno in all_annot:
            vid, show_name = anno.split('/')[-3], anno.split('/')[-2]
            df = pd.read_csv(anno, header=None, index_col=False,
                         names=['path', 'xmin', 'ymin', 'xmax', 'ymax', 'gazex', 'gazey'])
            values = df.values
            start_idx = 0
            while start_idx<len(values):
                frame_idx = torch.ones(seq_len)
                if start_idx + seq_len > len(values):
                    pad_num = start_idx + seq_len - len(values)
                    start_idx -= pad_num
                    frame_idx[:pad_num] = 0
                end_idx = start_idx+seq_len
                anno_seq.append((vid, show_name, frame_idx, df.iloc[start_idx:end_idx]))
                num_seq += 1
                num_frame += end_idx - start_idx
                start_idx += seq_len
        print(f"Mode: {mode}, num of sequences: {num_seq}, num of frames: {num_frame}")
        with open(os.path.join(anno_dir, mode, 'anno_seq.pickle'), 'wb') as file:
            pickle.dump(anno_seq, file)


