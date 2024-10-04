import os
import cv2
import torch
import pickle
import imagesize
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    # change to the overlap ratio with the groundtruth box
    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.t())
    area2 = box_area(box2.t())

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / area1[:, None]  # iou = inter / (area1 + area2 - inter)





base_dir = '/data/add_disk0/qiaomu/datasets/gaze/gazefollow'
img_dir = os.path.join(base_dir, 'train')
head_dets = os.path.join(base_dir, 'head_detections', 'head_dets.pkl')
with open(head_dets, 'rb') as file:
    all_dets = pickle.load(file)

ori_annt = os.path.join(base_dir, "train_annotations_release_persondet.txt")
column_names = ['path', 'idx', 'body_bbox_x', 'body_bbox_y', 'body_bbox_w', 'body_bbox_h', 'eye_x', 'eye_y',
                            'gaze_x', 'gaze_y', 'bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max', 'inout', 'meta', 'ori_name',
                            'body_x1', 'body_y1', 'body_x2', 'body_y2']

df = pd.read_csv(ori_annt, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
df = df[df['inout'] != -1]
iou_thres = 0.7
length_thres = 35 # boxes with height/width ratio less than this will be discarded
ratio_thres = 0.05

df_unsup = []
new_column_names = ['path', 'bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max']
sample_num = 100
np.random.seed(0)
perm_indices = np.random.permutation(range(df.shape[0]))[:sample_num]
save_folder = os.path.join(base_dir, "head_detections", "examples")

for idx_df, row in tqdm(df.iterrows()):
    #if idx_df not in perm_indices:
    #    continue
    path, box_x1, box_x2, box_y1, box_y2 = row['path'], row['bbox_x_min'], row['bbox_x_max'], row['bbox_y_min'], row['bbox_y_max']
    dirname = os.path.dirname(path)
    imgname = os.path.basename(path)
    this_dets = all_dets[imgname]
    if len(this_dets)<=1:
        continue
    ori_head = torch.tensor([[box_x1, box_y1, box_x2, box_y2]]).int()
    this_dets = torch.tensor(this_dets).squeeze(1).tolist()
    iou = box_iou(ori_head, torch.tensor(this_dets).int())[0] # N
    img_path = os.path.join(base_dir, path)
    #img = cv2.imread(img_path)
    width, height = imagesize.get(img_path)
    start_pt, end_pt = (int(box_x1), int(box_y1)), (int(box_x2), int(box_y2))
    #cv2.rectangle(img, start_pt, end_pt, (0,255,0), thickness=2)
    count_idx=0
    for idx, det in enumerate(this_dets):
        if iou[idx] >= iou_thres:
            continue
        
        det_x1, det_y1, det_x2, det_y2 = det
        start_pt, end_pt = (int(det_x1), int(det_y1)), (int(det_x2), int(det_y2))
        box_h, box_w = det_y2 - det_y1, det_x2 - det_x1
        #if box_h < length_thres or box_w < length_thres:
        #    continue
        if box_h / height < ratio_thres or box_w / width < ratio_thres or box_h < length_thres or box_w < length_thres:
            continue
        df_this = [os.path.join(dirname, imgname), *det]
        df_unsup.append(df_this)
        #cv2.rectangle(img, start_pt, end_pt, (0,0,255), thickness=2)
        count_idx += 1
        
    #if count_idx > 0:
        #cv2.imwrite(os.path.join(save_folder, imgname), img)
    

print(f"Unlabelled samples: {len(df_unsup)}")
df_unsup = pd.DataFrame(df_unsup, columns=new_column_names)
df_unsup.to_csv(os.path.join(base_dir, "weak_supervision", "train_annotations_unsup_sampled.txt"), header=False, index=False)