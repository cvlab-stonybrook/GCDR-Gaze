import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
import pickle
import pdb
from tqdm import tqdm
import numpy as np
from torchvision import transforms
from config import * 
from utils import imutils, evaluation, misc
from utils.soft_argmax import softargmax2d
from datasets.dataset_gradcam import GazeFollow_body_head
from skimage.transform import resize


def _get_transform():
    transform_list = []
    transform_list.append(transforms.Resize((input_resolution, input_resolution)))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(transform_list)


transform = _get_transform()
test_dataset = GazeFollow_body_head(gazefollow_val_data, transform, test=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=16,
                                            shuffle=False,
                                            num_workers=8)


AUC_multihot = {'9th':[], '10th':[], '11th':[]}
AUC_heatmap = {'9th':[], '10th':[], '11th':[]}
num_batches = len(test_loader)

min_dist = {'9th':[], '10th':[], '11th':[]}; avg_dist = {'9th':[], '10th':[], '11th':[]}
use_coord_regress=False
gradcam_size = 16


for batch_idx, data_input in enumerate(tqdm(test_loader)):
    image, img_path, eye_x, cont_gaze, gaze_heatmap_merge, gradcam_9th, gradcam_10th, gradcam_11th, imsize = data_input
    #gaze_heatmap_merge = gaze_heatmap_merge.numpy()
    
    gradcam_9th = gradcam_9th.unsqueeze(1)  # channel layer
    gradcam_10th = gradcam_10th.unsqueeze(1)  # channel layer
    gradcam_11th = gradcam_11th.unsqueeze(1)  # channel layer
    
    for b_i in range(len(cont_gaze)):
        this_path, this_eyex = img_path[b_i], eye_x[b_i]
        multi_hot = imutils.multi_hot_targets(cont_gaze[b_i], imsize[b_i])

        #scaled_heatmap = resize(gaze_heatmap_merge[b_i], (imsize[b_i][1], imsize[b_i][0]))
        #multi_hot = (gaze_heatmap_merge[b_i] > 0).float() 
        #multi_hot = misc.to_numpy(multi_hot)
        
        valid_gaze = cont_gaze[b_i]
        valid_gaze = valid_gaze[valid_gaze != -1].view(-1,2)
        
        width, height = imsize[b_i]
        #width, height = 64, 64
        gradcam_9th_this, gradcam_10th_this, gradcam_11th_this = gradcam_9th[b_i].squeeze(), gradcam_10th[b_i].squeeze(), gradcam_11th[b_i].squeeze()
        gradcam_9th_resize = torch.nn.functional.interpolate(gradcam_9th[b_i:b_i+1], size=(height, width), mode='bilinear', align_corners=False).squeeze()
        gradcam_10th_resize = torch.nn.functional.interpolate(gradcam_10th[b_i:b_i+1], size=(height, width), mode='bilinear', align_corners=False).squeeze()
        gradcam_11th_resize = torch.nn.functional.interpolate(gradcam_11th[b_i:b_i+1], size=(height, width), mode='bilinear', align_corners=False).squeeze()

        gradcam_9th_resize = (gradcam_9th_resize - gradcam_9th_resize.min()) / (gradcam_9th_resize.max() - gradcam_9th_resize.min())
        gradcam_10th_resize = (gradcam_10th_resize - gradcam_10th_resize.min()) / (gradcam_10th_resize.max() - gradcam_10th_resize.min())
        gradcam_11th_resize = (gradcam_11th_resize - gradcam_11th_resize.min()) / (gradcam_11th_resize.max() - gradcam_11th_resize.min())
        gradcam_9th_resize, gradcam_10th_resize, gradcam_11th_resize = gradcam_9th_resize.numpy(), gradcam_10th_resize.numpy(), gradcam_11th_resize.numpy()
        
        
        if np.isnan(gradcam_9th_resize).any() or np.isnan(gradcam_10th_resize).any() or np.isnan(gradcam_11th_resize).any():
            print("NANs in {}".format(this_path))
            continue
        
        
        auc_9th = evaluation.auc(gradcam_9th_resize, multi_hot)
        auc_10th = evaluation.auc(gradcam_10th_resize, multi_hot)
        auc_11th = evaluation.auc(gradcam_11th_resize, multi_hot)
        #print(auc_10th)
        #print(auc_11th)
        AUC_multihot['9th'].append(auc_9th)
        AUC_multihot['10th'].append(auc_10th)
        AUC_multihot['11th'].append(auc_11th)
        
        
        
        pred_x_9, pred_y_9 = evaluation.argmax_pts(gradcam_9th_this)
        pred_x_10, pred_y_10 = evaluation.argmax_pts(gradcam_10th_this)
        pred_x_11, pred_y_11 = evaluation.argmax_pts(gradcam_11th_this)
            
        norm_p_9 = [pred_x_9/float(gradcam_size), pred_y_9/float(gradcam_size)]
        norm_p_10 = [pred_x_10/float(gradcam_size), pred_y_10/float(gradcam_size)]
        norm_p_11 = [pred_x_11/float(gradcam_size), pred_y_11/float(gradcam_size)]
        all_distances_9, all_distances_10, all_distances_11 = [], [], []
        for gt_gaze in valid_gaze:
            all_distances_9.append(evaluation.L2_dist(gt_gaze, norm_p_9))
            all_distances_10.append(evaluation.L2_dist(gt_gaze, norm_p_10))
            all_distances_11.append(evaluation.L2_dist(gt_gaze, norm_p_11))
        
        min_dist['9th'].append(min(all_distances_9))
        min_dist['10th'].append(min(all_distances_10))
        min_dist['11th'].append(min(all_distances_11))
        # average distance: distance between the predicted point and human average point
        mean_gt_gaze = torch.mean(valid_gaze, 0)
        avg_distance_9 = evaluation.L2_dist(mean_gt_gaze, norm_p_9)
        avg_distance_10 = evaluation.L2_dist(mean_gt_gaze, norm_p_10)
        avg_distance_11 = evaluation.L2_dist(mean_gt_gaze, norm_p_11)
        avg_dist['9th'].append(avg_distance_9)
        avg_dist['10th'].append(avg_distance_10)
        avg_dist['11th'].append(avg_distance_11)
    
    print("Batch {}/{} AUC 9th layer: {} AUC 10th layer: {}, AUC 11th layer: {} \n Avg dist 9th: {} 10th: {} 11th: {}, Min dist 9th:{} 10th: {}, 11th: {}".format(
        batch_idx, num_batches, torch.mean(torch.tensor(AUC_multihot['9th'])), torch.mean(torch.tensor(AUC_multihot['10th'])),torch.mean(torch.tensor(AUC_multihot['11th'])),
        torch.mean(torch.tensor(avg_dist['9th'])), torch.mean(torch.tensor(avg_dist['10th'])), torch.mean(torch.tensor(avg_dist['11th'])),
        torch.mean(torch.tensor(min_dist['9th'])), torch.mean(torch.tensor(min_dist['10th'])), torch.mean(torch.tensor(min_dist['11th']))))     

print("AUC 9th layer: {} 10th layer: {}, AUC 11th layer: {}, \n Avg dist 9th: {}  10th: {} 11th: {}, Min dist 9th: {}  10th: {}, 11th: {}".format(
    torch.mean(torch.tensor(AUC_multihot['9th'])), torch.mean(torch.tensor(AUC_multihot['10th'])),torch.mean(torch.tensor(AUC_multihot['11th'])),
    torch.mean(torch.tensor(avg_dist['9th'])), torch.mean(torch.tensor(avg_dist['10th'])), torch.mean(torch.tensor(avg_dist['11th'])),
    torch.mean(torch.tensor(min_dist['9th'])), torch.mean(torch.tensor(min_dist['10th'])), torch.mean(torch.tensor(min_dist['11th']))))
