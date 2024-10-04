import torch, detectron2
from detectron2.utils.logger import setup_logger
from torch.utils.data import DataLoader
import glob
# import some common libraries
import numpy as np
import pandas as pd
import os, json, cv2, random
import yaml, pickle
import pdb
import traceback
from tqdm import tqdm
from matplotlib import pyplot as plt
from easydict import EasyDict as edict
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.detection_utils import read_image
import matplotlib.patches as patches



def collate_fn(batch):
    return batch

def get_random_color():
    r = random.randint(0, 255) / 255
    g = random.randint(0, 255) / 255
    b = random.randint(0, 255) / 255
    return (r,g,b)


data_split = 'test'
config_file = './config/config_transformer.yaml'
with open(config_file) as f:
    cfg_dict = yaml.load(f, Loader=yaml.FullLoader)

data_dir = '/nfs/bigrod/add_disk0/qiaomu/datasets/gaze/videoattentiontarget'
imgout_dir = os.path.join(data_dir, 'vis_detections')       

image_dir = os.path.join(data_dir, 'images')
shows = glob.glob(os.path.join(data_dir, 'annotations', data_split, '*'))
all_sequence_paths = []
for s in shows:
    sequence_annotations = glob.glob(os.path.join(s, '*', '*.txt'))  # each annotation file is for one person
    all_sequence_paths.extend(sequence_annotations)

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)
stat_write_dir = os.path.join(data_dir, 'person_detections')
if not os.path.exists(stat_write_dir):
    os.makedirs(stat_write_dir)
img_paths = set()
persondet_info = {}
alldet_info = {}
num_duplicate = 0
try:
    for idx, sequence_path in enumerate(tqdm(all_sequence_paths)):
        df = pd.read_csv(sequence_path, header=None, index_col=False,
                        names=['path', 'xmin', 'ymin', 'xmax', 'ymax', 'gazex', 'gazey'])
        person_name = sequence_path.split('/')[-1].split('.')[0]
        show_name = sequence_path.split('/')[-3]
        clip = sequence_path.split('/')[-2]
        seq_len = len(df.index)

        for i, row in df.iterrows():
            img_name, head_x1, head_y1, head_x2, head_y2, gaze_x, gaze_y = row[['path', 'xmin', 'ymin', 'xmax', 'ymax', 'gazex', 'gazey']]
            img_path = os.path.join(image_dir, show_name, clip, img_name)
            imgname = img_name.split('.')[0]
            img_name = img_path.split("/")[-1].split('.')[0]
            
            if img_path in img_paths:
                print(f"{img_path} duplicate!")
                num_duplicate+=1
                continue
            img_paths.add(img_path)
            
            img = read_image(img_path, format="BGR")
            outputs = predictor(img)
            pred_classes = outputs['instances'].pred_classes.cpu().numpy()
            pred_boxes = outputs['instances'].pred_boxes.tensor.cpu().numpy()
            pred_scores = outputs['instances'].scores.cpu().numpy()
            idx = pred_classes==0
            person_boxes = pred_boxes[idx]
            person_scores = pred_scores[idx]
            #v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            #out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            persondet_info[img_path]={'boxes': person_boxes, 'num_person':len(person_boxes), 'scores':person_scores}
            alldet_info[img_path]={'boxes': pred_boxes, 'classes':pred_classes, 'scores':pred_scores}
            
            #cv2.imwrite(os.path.join(imgout_dir, img_name+'.jpg'), img)
             
            #out.save(os.path.join(imgout_dir, img_name +"_det.jpg"))
            
            '''
            fig = plt.figure(figsize=(12,8))
            plt.subplot(1,2,1)
            ax = plt.gca()
            ax.axis('off')
            detectron_out = cv2.imread(os.path.join(imgout_dir, img_name +"_det.jpg"))[:,:,::-1] 
            img = img[:,:,::-1] 
            plt.imshow(detectron_out) 
            plt.subplot(1,2,2)
            plt.imshow(img)
            height,width,_=img.shape
            ax.axis('off')
            for idx in range(len(bboxes)):
                bbox, head_box = bboxes[idx], head_boxes[idx]
                bbox[0] = np.clip(bbox[0], 0, width-1)
                bbox[2] = np.clip(bbox[2], 0, width-1)
                bbox[1] = np.clip(bbox[1], 0, height-1)
                bbox[3] = np.clip(bbox[3], 0, height-1)
                color = get_random_color()
                rect = patches.Rectangle((int(bbox[0]), int(bbox[1])), int(bbox[2]-bbox[0]), int(bbox[3]-bbox[1]), linewidth=3, edgecolor=color, facecolor='none')     
                ax = plt.gca()
                ax.add_patch(rect)
                rect = patches.Rectangle((int(head_box[0]), int(head_box[1])), int(head_box[2]-head_box[0]), int(head_box[3]-head_box[1]), linewidth=3, edgecolor=color, facecolor='none')     
                ax.add_patch(rect)
            plt.savefig(os.path.join(imgout_dir, f'{img_name}_comp.jpg'))
            plt.close()
            '''

    with open(os.path.join(stat_write_dir, f'person_detections_{data_split}.pkl'), 'wb') as file:
        pickle.dump(persondet_info, file)
    with open(os.path.join(stat_write_dir, f'all_detections_{data_split}.pkl'), 'wb') as file:
        pickle.dump(alldet_info, file)
    print("Number of duplicates: {}".format(num_duplicate))  # duplicates: 6368

except Exception:
    print(traceback.format_exc())
   