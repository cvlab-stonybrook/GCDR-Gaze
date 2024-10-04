from __future__ import absolute_import
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from skimage.transform import resize

def auc(heatmap, onehot_im, is_im=True):
    if is_im:
        auc_score = roc_auc_score(np.reshape(onehot_im,onehot_im.size), np.reshape(heatmap,heatmap.size))
    else:
        auc_score = roc_auc_score(onehot_im, heatmap)
    return auc_score


def ap(label, pred):
    return average_precision_score(label, pred)


def argmax_pts(heatmap):
    idx = np.unravel_index(heatmap.argmax(), heatmap.shape)
    pred_y, pred_x = map(float,idx)
    return pred_x, pred_y


def L2_dist(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def get_all_metrics(gaze_heatmap_pred, multi_hot, valid_gaze, img_width, img_height, output_resolution=64):

    scaled_heatmap = resize(gaze_heatmap_pred, (img_width, img_height))
    auc_score = auc(scaled_heatmap, multi_hot)
    # min distance: minimum among all possible pairs of <ground truth point, predicted point>
    pred_x, pred_y = argmax_pts(gaze_heatmap_pred)
    norm_p = [pred_x/float(output_resolution), pred_y/float(output_resolution)]
    all_distances = []
    for gt_gaze in valid_gaze:
        all_distances.append(L2_dist(gt_gaze, norm_p))
    min_dist = min(all_distances)
    # average distance: distance between the predicted point and human average point
    mean_gt_gaze = np.mean(valid_gaze, axis=0)
    avg_distance = L2_dist(mean_gt_gaze, norm_p) 
    
    return auc_score, avg_distance, min_dist
