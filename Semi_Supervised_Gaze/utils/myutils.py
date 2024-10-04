import matplotlib as mpl
mpl.use('Agg')
import cv2
from matplotlib import pyplot as plt
from easydict import EasyDict as edict
import yaml
import numpy as np
import pandas as pd
from config import *
import logging
import torch
import os, pickle
import warnings
import shutil
import torch.nn.functional as F
from PIL import Image
from tensorboardX import SummaryWriter
import matplotlib.patches as patches
from utils import imutils, evaluation, misc
from torchvision import transforms
from collections import OrderedDict
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix,average_precision_score
from skimage.transform import resize

def update_config(config_file):
    with open(config_file) as f:
        config = edict(yaml.load(f, Loader=yaml.FullLoader))
        return config




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

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.t())
    area2 = box_area(box2.t())

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def get_logger(log_file):
    
    logger = logging.getLogger("logger")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s- %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

def get_transform(input_resolution):
    transform_list = []
    transform_list.append(transforms.Resize((input_resolution, input_resolution)))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(transform_list)

def get_patch_dist_from_hm(gaze_heatmap, inout_label, hm_size=64, patch_num=7):
   
    patch_size = hm_size // patch_num  # modify here
    steps = patch_num
    bs = gaze_heatmap.shape[0]
    patch_dist = []
    for b_i in range(bs):
        inout_patch = []
        for i in range(steps):
            for j in range(steps):
                inout_patch.append(gaze_heatmap[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size].max())
        patch_dist.append(torch.tensor(inout_patch).to(gaze_heatmap))
    patch_dist = torch.stack(patch_dist)
    
    out_prob = (torch.ones(inout_label.size()).to(inout_label) - inout_label).unsqueeze(1)
    patch_dist = torch.cat((patch_dist, out_prob), dim=-1)
    patch_dist = patch_dist / patch_dist.sum(dim=1, keepdim=True)
        
    return patch_dist


def setup_logger_tensorboard(project_name, setting_name, ckpt_base_dir, resume=-1):
    logdir = os.path.join('./logs', project_name, setting_name)
    ckpt_dir = os.path.join(ckpt_base_dir, project_name, setting_name)
    if resume==-1 and os.path.exists(logdir):
        shutil.rmtree(logdir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    writer = SummaryWriter(logdir)
    log_path = os.path.join(logdir, "{}.log".format('train'))
    logger = get_logger(log_path)
    return logdir, ckpt_dir, logger, writer


def smooth_by_conv(window_size, df, col):
    padded_track = pd.concat([pd.DataFrame([[df.iloc[0][col]]]*(window_size//2), columns=[0]),
                     df[col],
                     pd.DataFrame([[df.iloc[-1][col]]]*(window_size//2), columns=[0])])
    smoothed_signals = np.convolve(padded_track.squeeze(), np.ones(window_size)/window_size, mode='valid')
    return smoothed_signals


# added by miao, evaluation with mAP
def evaluation_ap(pred_logits, target_labels, logger, label_names, mode, epoch):
    # calculate precision, recall, F1, and AP for each category
    pred_labels = np.argmax(pred_logits, axis=1)
    accuracy = (pred_labels==target_labels).sum() / len(target_labels)
    logger.info("Top 1 Precision:")
    for i in range(len(label_names)):
        num_this_real = np.sum(target_labels==i)
        num_this_pred = np.sum(pred_labels==i)
        num_correct_recall = np.sum(pred_labels[target_labels==i]==i)
        num_correct_precision = np.sum(target_labels[pred_labels==i]==i)
        category = label_names[i]
        recall = num_correct_recall/num_this_real if num_this_real!=0 else 0
        precision = num_correct_precision/num_this_pred if num_this_pred!=0 else 0
        F1 = 2*precision*recall / (precision+recall) if precision+recall!=0 else 0
        logger.info("Mode:{} Recall: Category {} True samples: {}, Acc: {}".format(mode, category, num_this_real, recall))
        logger.info("Mode:{} Precision: Category {} Predict samples: {}, Acc: {}".format(mode, category, num_this_pred, precision))
        logger.info("Mode:{} F1: Category {}: {}".format(mode, category, F1))
    logger.info("Average Precision:")
    mAP = 0.0
   
    for i in range(len(label_names)):
        one_hot = (target_labels==i).astype(int)
        logits_input = pred_logits[:,i]
        ap_this = average_precision_score(one_hot, logits_input) if one_hot.sum()>0 else 0
        mAP+=ap_this
        category = label_names[i]
        logger.info("Mode:{}: Category {}, Ap: {}".format(mode, category, ap_this))
    mAP /= len(label_names)
    
    logger.info("Epoch {} Mode:{} Top-1 Accuracy: {}, mAP:{}".format(epoch, mode, accuracy, mAP))
    logger.info("\n")
    return mAP


def load_checkpoint(fpath):
    r"""Loads checkpoint.

    ``UnicodeDecodeError`` can be well handled, which means
    python2-saved files can be read from python3.

    Args:
        fpath (str): path to checkpoint.
    
    Returns:
        dict

    Examples::
        >>> from torchreid.utils import load_checkpoint
        >>> fpath = 'log/my_model/model.pth.tar-10'
        >>> checkpoint = load_checkpoint(fpath)
    """
    if fpath is None:
        raise ValueError('File path is None')
    if not os.path.exists(fpath):
        raise FileNotFoundError('File is not found at "{}"'.format(fpath))
    map_location = None if torch.cuda.is_available() else 'cpu'
    try:
        checkpoint = torch.load(fpath, map_location=map_location)
    except UnicodeDecodeError:
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        checkpoint = torch.load(
            fpath, pickle_module=pickle, map_location=map_location
        )
    except Exception:
        print('Unable to load checkpoint from "{}"'.format(fpath))
        raise
    return checkpoint


def load_pretrained_weights(model, state_dict = None, weight_path=None):
    r"""Loads pretrianed weights to model.

    Features::
        - Incompatible layers (unmatched in name or size) will be ignored.
        - Can automatically deal with keys containing "module.".

    Args:
        model (nn.Module): network model.
        weight_path (str): path to pretrained weights.

    Examples::
        >>> from torchreid.utils import load_pretrained_weights
        >>> weight_path = 'log/my_model/model-best.pth.tar'
        >>> load_pretrained_weights(model, weight_path)
    """
    if state_dict is None:
        checkpoint = load_checkpoint(weight_path)
        state_dict = checkpoint

    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    elif 'model' in state_dict:
        state_dict = state_dict['model']
        
    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:] # discard module.

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    
    if len(matched_layers) == 0:
        warnings.warn(
            'The pretrained weights "{}" cannot be loaded, '
            'please check the key names manually '
            '(** ignored and continue **)'.format(weight_path)
        )
    else:
        print(
            'Successfully loaded pretrained weights from "{}"'.
            format(weight_path)
        )
        if len(discarded_layers) > 0:
            print(
                '** The following layers are discarded '
                'due to unmatched keys or layer size: {}'.
                format(discarded_layers)
            )

def load_submodule_weights(model, module_name, state_dict=None, weight_path=None):
    # module name: the name of the submodule in the main model
    
    if state_dict is None:
        checkpoint = load_checkpoint(weight_path)
        state_dict = checkpoint
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    elif 'model' in state_dict:
        state_dict = state_dict['model']
    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:] # discard module.

        if k.startswith(module_name):
            k = k[len(module_name)+1:]
            if k in model_dict and model_dict[k].size() == v.size():
                new_state_dict[k] = v
                matched_layers.append(k)
            else:
                discarded_layers.append(k)
    
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    
    if len(matched_layers) == 0:
        warnings.warn(
            'The pretrained weights "{}" cannot be loaded, '
            'please check the key names manually '
            '(** ignored and continue **)'.format(weight_path)
        )
    else:
        print(
            'Successfully loaded pretrained weights from "{}" for {}'.
            format(weight_path, module_name)
        )
        if len(discarded_layers) > 0:
            print(
                '** The following layers are discarded '
                'due to unmatched keys or layer size: {}'.
                format(discarded_layers)
            )     


def get_entropy_from_hm(hm, eps=1e-11):
    
    is_numpy = False
    if type(hm) is not torch.Tensor:
        is_numpy = True
        hm = torch.tensor(hm).float()
    hm = hm.clone() 
    hm[hm==0] = eps
    # suppose hm is in the size of (bs, hm_h, hm_w)
    if len(hm.size())==2:
        hm = hm.unsqueeze(0)
    hm = hm.flatten(start_dim=1)
    entropy = -1 * (hm * torch.log(hm)).sum(dim=1)
    entropy = entropy.squeeze()
    if is_numpy:
        entropy = np.array(entropy)
    
    return entropy


def get_eyemaps_facealignment(fa, face_A_imgs, face_B_imgs, eyemap_coarse_A, eyemap_coarse_B, head_coords_A, head_coords_B, heatmap_size=64, thres=0.5):
    eyemap_A, eyemap_B = [],[]
    eye_kp_A, eye_kp_B = [],[]
    bs, h, w, _ = face_A_imgs.shape
    not_detected, low_score = 0,0
    head_coords_A, head_coords_B = head_coords_A.numpy(), head_coords_B.numpy()
    for i in range(bs):
        for j in range(2):
            face = face_A_imgs[i] if j==0 else face_B_imgs[i]
            landmarks, landmark_scores, detected_faces = fa.get_landmarks_from_image(face, detected_faces=None, return_landmark_score=True)
            eyemap = eyemap_A if j==0 else eyemap_B
            eyemap_coarse = eyemap_coarse_A[i] if j==0 else eyemap_coarse_B[i]
            head_coords = head_coords_A[i] if j==0 else head_coords_B[i]
            eye_kp = eye_kp_A if j==0 else eye_kp_B
            idx = 'A' if j==0 else 'B'

            if landmarks is None or len(landmarks)==0:
                not_detected+=1
                eyemap.append(eyemap_coarse)
                eye_kp.append(None)
            else:
                keypoints, scores = landmarks[0].copy(), landmark_scores[0]
                #print(f"image {i} left eye scores: {scores[36:42]} right eye scores: {scores[42:48]}")
                leftscore, rightscore = scores[36:42].mean(), scores[42:48].mean()
                if (leftscore+rightscore) / 2 <thres:
                    #print(f"image {i} detection score too low!")
                    low_score+=1
                    eyemap.append(eyemap_coarse)
                    eye_kp.append(None)
                    continue
                keypoints[:,0] /= w
                keypoints[:,1] /= h
                keypoints[:, 0] = head_coords[0] + (keypoints[:, 0] * (head_coords[2]-head_coords[0]))
                keypoints[:, 1] = head_coords[1] + (keypoints[:, 1] * (head_coords[3]-head_coords[1]))
                keypoints[:,0] *= 224
                keypoints[:,1] *= 224
                keypoints = keypoints.astype(int)
                eye_kp.append(keypoints)
                lefteye = landmarks[0][36:42].mean(axis=0)
                righteye = landmarks[0][42:48].mean(axis=0)
                mid_point = (lefteye+righteye)/2
                mid_point[0] = mid_point[0]/w
                mid_point[1] = mid_point[1]/h
                eye_x, eye_y = mid_point[0]*(head_coords[2]-head_coords[0])+head_coords[0], mid_point[1]*(head_coords[3]-head_coords[1])+head_coords[1]
                eye_heatmap = torch.zeros(heatmap_size, heatmap_size)  # set the size of the output
                eyemap_this = imutils.draw_labelmap(eye_heatmap.clone(), [eye_x * heatmap_size, eye_y * heatmap_size],3,type='Gaussian')
                eyemap.append(eyemap_this)
    eyemap_A, eyemap_B = torch.stack(eyemap_A), torch.stack(eyemap_B)
    if not_detected!=0:
        print("Number of {} faces not detected!".format(not_detected))
    print("Number of {} faces has low score!".format(low_score))
    return eyemap_A, eyemap_B, eye_kp_A, eye_kp_B


def get_eyemaps_mmpose(pose_model_info, face_A_imgs, face_B_imgs, eyemap_coarse_A, eyemap_coarse_B, head_coords_A, head_coords_B, heatmap_size=64, thres=0.5):
    from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)
    from mmpose.datasets import DatasetInfo
    
    eyemap_A, eyemap_B = [],[]
    eye_kp_A, eye_kp_B = [],[]
    bs, h, w, _ = face_A_imgs.shape
    not_detected, low_score = 0,0
    pose_model, dataset, dataset_info = pose_model_info['model'], pose_model_info['dataset'], pose_model_info['dataset_info']
    for i in range(bs):
        for j, idx in enumerate(['A','B']):
            face_img = face_A_imgs[i] if j==0 else face_B_imgs[i]
            #face_img = face_img.astype(np.float32)
            face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
            #face_img = face_img[:,:,::-1]
            person_results = [{'bbox': np.array([0, 0, w, h])}]
            pose_results, returned_outputs = inference_top_down_pose_model(
                pose_model,
                face_img,
                person_results,
                format='xyxy',
                dataset=dataset,
                dataset_info=dataset_info,
                return_heatmap=False,
                outputs=None)
            eyemap = eyemap_A if j==0 else eyemap_B
            eyemap_coarse = eyemap_coarse_A[i] if j==0 else eyemap_coarse_B[i]
            head_coords = head_coords_A[i] if j==0 else head_coords_B[i]
            eye_kp = eye_kp_A if j==0 else eye_kp_B

            if len(pose_results)==0:
                print("image {} not detected!".format(i))
                not_detected+=1
                eyemap.append(eyemap_coarse)
                eye_kp.append(None)
            else:
                keypoints, scores = pose_results[0]['keypoints'][:, :2], pose_results[0]['keypoints'][:, -1]
                leftscore, rightscore = scores[36:42].mean(), scores[42:48].mean()
                #if (leftscore<thres or rightscore<thres):
                if (leftscore+rightscore) / 2 <thres:
                    #print(f"image {i}, left: {scores[36:42]}")
                    #print(f"image {i}, right: {scores[42:48]}")
                    #print(f"image {i} detection score too low!")
                    low_score+=1
                    eyemap.append(eyemap_coarse)
                    eye_kp.append(None)
                    continue
                keypoints[:,0] /= w
                keypoints[:,1] /= h
                keypoints[:, 0] = head_coords[0] + (keypoints[:, 0] * (head_coords[2]-head_coords[0]))
                keypoints[:, 1] = head_coords[1] + (keypoints[:, 1] * (head_coords[3]-head_coords[1]))
                
                eye_kp.append((keypoints*224).astype(int))
                lefteye, righteye = keypoints[36:42].mean(axis=0), keypoints[42:48].mean(axis=0)
                mid_point = (lefteye+righteye)/2
                eye_x, eye_y = mid_point[0], mid_point[1]
                eye_heatmap = torch.zeros(heatmap_size, heatmap_size)  # set the size of the output
                eyemap_this = imutils.draw_labelmap(eye_heatmap.clone(), [eye_x * heatmap_size, eye_y * heatmap_size],3,type='Gaussian')
                eyemap.append(eyemap_this)
    eyemap_A, eyemap_B = torch.stack(eyemap_A), torch.stack(eyemap_B)
    if not_detected!=0:
        print("Number of {} faces not detected!".format(not_detected))
    print("Number of {} faces has low score!".format(low_score))
    return eyemap_A, eyemap_B, eye_kp_A, eye_kp_B

def eval_auc_sample(gaze_coords, inout_label, gaze_heatmap_pred, AUC, distance):
    for b_i in range(len(inout_label)):
        # remove padding and recover valid ground truth points
        valid_gaze = gaze_coords[b_i]
        valid_gaze = valid_gaze[valid_gaze != -1].reshape(-1,2)
        # AUC: area under curve of ROC
        if inout_label[b_i]:
                # AUC: area under curve of ROC
            multi_hot = torch.zeros(output_resolution, output_resolution)  # set the size of the output
            gaze_x = gaze_coords[b_i, 0]
            gaze_y = gaze_coords[b_i, 1]

            multi_hot = imutils.draw_labelmap(multi_hot, [gaze_x * output_resolution, gaze_y * output_resolution], 3, type='Gaussian')
            multi_hot = (multi_hot > 0).float() * 1 # make GT heatmap as binary labels
            multi_hot = misc.to_numpy(multi_hot)

            scaled_heatmap = resize(gaze_heatmap_pred[b_i].squeeze(), (output_resolution, output_resolution))
            auc_score = evaluation.auc(scaled_heatmap, multi_hot)
            AUC.append(auc_score)

            # distance: L2 distance between ground truth and argmax point
            pred_x, pred_y = evaluation.argmax_pts(gaze_heatmap_pred[b_i])
 
            norm_p = [pred_x/output_resolution, pred_y/output_resolution]
            dist_score = evaluation.L2_dist(gaze_coords[b_i], norm_p).item()
            distance.append(dist_score)


def plot_hm_with_img(deconv, inout_val, head_coords, imsizes, img_paths, save_img_dir, att_save_dir=None, attn_weights=None, plot_dpi=300, vis_mode='heatmap', out_threshold=0.5):
    # all inputs are in the form of a batch
    bs = deconv.shape[0] 

    for b_i in range(bs): 
        ori_size = imsizes[b_i].numpy()
        ori_width, ori_height = ori_size[0], ori_size[1]
        head_box = head_coords[b_i].numpy()
        head_box[0], head_box[2] = head_box[0] * ori_width, head_box[2] * ori_width
        head_box[1], head_box[3] = head_box[1] * ori_height, head_box[3] * ori_height
        img_path_this = img_paths[b_i]
        #import pdb
        #pdb.set_trace()
        show_name, imgname = img_path_this.split('/')[-3], img_path_this.split('/')[-1].split('.')[0]
        save_img_dir_this = os.path.join(save_img_dir, show_name)
    
        if not os.path.exists(save_img_dir_this):
            os.makedirs(save_img_dir_this)
        if att_save_dir is not None:
            save_att_dir_this = os.path.join(att_save_dir, show_name)
            if not os.path.exists(save_att_dir_this):
                os.makedirs(save_att_dir_this)

        raw_hm = deconv[b_i] * 255
        raw_hm = raw_hm.squeeze()
        inout = inout_val[b_i,0].item()
        inout = 1 / (1 + np.exp(-inout)) # sigmoid
        inout = (1 - inout) * 255
        #norm_map = resize(raw_hm, (ori_height, ori_width)) - inout
        norm_map = resize(raw_hm, (ori_height, ori_width)) # not modify with inout, see how it goes
        fig = plt.figure(figsize=(ori_width/plot_dpi, ori_height/plot_dpi), dpi=plot_dpi)
        plt.axis('off')
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        frame_raw = cv2.imread(img_path_this)[:,:,::-1]
        assert (ori_height, ori_width) == (frame_raw.shape[0], frame_raw.shape[1]), "{},{};;{},{}".format(ori_height, ori_width, frame_raw.shape[0]. frame_raw.shape[1])
        plt.imshow(frame_raw)
        ax = plt.gca()
        rect = patches.Rectangle((int(head_box[0]), int(head_box[1])), int(head_box[2]-head_box[0]), int(head_box[3]-head_box[1]), linewidth=2, edgecolor=(0,1,0), facecolor='none')
        ax.add_patch(rect)

        if vis_mode == 'arrow':
            if inout < out_threshold: # in-frame gaze
                pred_x, pred_y = evaluation.argmax_pts(raw_hm)
                norm_p = [pred_x/output_resolution, pred_y/output_resolution]
                circ = patches.Circle((norm_p[0]*ori_width, norm_p[1]*ori_height), ori_height/50.0, facecolor=(0,1,0), edgecolor='none')
                ax.add_patch(circ)
                plt.plot((norm_p[0]*ori_width,(head_box[0]+head_box[2])/2), (norm_p[1]*ori_height,(head_box[1]+head_box[3])/2), '-', color=(0,1,0,1))
        else:
            plt.imshow(norm_map, cmap = 'jet', alpha=0.2, vmin=0, vmax=255)
        pred_inout = np.around(inout_val[b_i,0].item(), decimals=2)
        plt.savefig(os.path.join(save_img_dir_this, "result_{}_predio{}.jpg".format(imgname, pred_inout)), pad_inches=0, bbox_inches='tight', dpi=plot_dpi)
        plt.close()

        if attn_weights is not None:
            # plot attention weights
            fig = plt.figure(figsize=(ori_width/plot_dpi, ori_height/plot_dpi), dpi=plot_dpi)
            plt.axis('off')
            plt.margins(0,0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.imshow(frame_raw)
            ax = plt.gca()
            rect = patches.Rectangle((int(head_box[0]), int(head_box[1])), int(head_box[2]-head_box[0]), int(head_box[3]-head_box[1]), linewidth=2, edgecolor=(0,1,0), facecolor='none')
            ax.add_patch(rect)
            norm_attmap = resize(attn_weights[b_i], (ori_height, ori_width), preserve_range=True) 
            
            #pdb.set_trace() 
            plt.imshow(norm_attmap, cmap = 'gist_gray', alpha=0.85, vmin=0.0, vmax=1.0)
            plt.savefig(os.path.join(save_att_dir_this, "result_{}_predio{}.jpg".format(imgname, pred_inout)), pad_inches=0, bbox_inches='tight', dpi=plot_dpi)
            plt.close()


def get_eye_keypoints(img, fa, idx_tensor, headpose_model, headpose_transformations, thres_kp=0.5, thres_pose=60):
    landmarks, landmark_scores, detected_faces = fa.get_landmarks_from_image(img, detected_faces=None, return_landmark_score=True)
    eye_x1_l, eye_y1_l, eye_x2_l, eye_y2_l = -1, -1, -1, -1
    eye_x1_r, eye_y1_r, eye_x2_r, eye_y2_r = -1, -1, -1, -1
    h,w = img.shape[:2]
    head_pose = []
    if landmarks is not None:
        face_box = fa.face_detector.detect_from_image(img.copy())[0]
        x1,y1,x2,y2 = map(int, face_box[:4])
        x1, x2 = max(x1,0), min(x2,w-1)
        y1, y2 = max(y1,0), min(y2,h-1)
        # estimate head pose
        img = img[y1:y2, x1:x2].copy()
        img = Image.fromarray(img)
        # Transform
        img = headpose_transformations(img)
        img = img.unsqueeze(0).cuda()
        with torch.no_grad():
            yaw, pitch, roll = headpose_model(img)
        yaw_predicted = F.softmax(yaw, dim=1)
        pitch_predicted = F.softmax(pitch, dim=1)
        roll_predicted = F.softmax(roll, dim=1)

        yaw_predicted = torch.sum(yaw_predicted * idx_tensor, dim=1) * 3 - 99
        pitch_predicted = torch.sum(pitch_predicted.data * idx_tensor, dim=1) * 3 - 99
        roll_predicted = torch.sum(roll_predicted.data * idx_tensor, dim=1) * 3 - 99
        yaw_this, pitch_this, roll_this = yaw_predicted.item(), pitch_predicted.item(), roll_predicted.item()
        head_pose = [yaw_this, pitch_this, roll_this]

    if landmarks is None or len(landmarks)==0:
        eyes = 'NA'
        face_box=None
        keypoints, scores = 'NA', 'NA'
    else:
        keypoints, scores = landmarks[0].copy(), landmark_scores[0]
        #print(f"image {i} left eye scores: {scores[36:42]} right eye scores: {scores[42:48]}")
        leftscore, rightscore = scores[36:42].mean(), scores[42:48].mean()
        lefteye = keypoints[36:42] if leftscore>thres_kp else 'NA'
        righteye = keypoints[42:48] if rightscore>thres_kp else 'NA'
        if leftscore>thres_kp and yaw_this<thres_pose:
            eye_x1_l, eye_y1_l, eye_x2_l, eye_y2_l = lefteye[:,0].min(), lefteye[:,1].min(), lefteye[:,0].max(), lefteye[:,1].max() 
            eye_width, eye_height = eye_x2_l-eye_x1_l, eye_y2_l-eye_y1_l
        if rightscore>thres_kp and yaw_this>-thres_pose:
            eye_x1_r, eye_y1_r, eye_x2_r, eye_y2_r = righteye[:,0].min(), righteye[:,1].min(), righteye[:,0].max(), righteye[:,1].max() 
            eye_width, eye_height = eye_x2_r-eye_x1_r, eye_y2_r-eye_y1_r
    lst_add = np.array(list(map(int, [eye_x1_l, eye_y1_l, eye_x2_l, eye_y2_l, eye_x1_r, eye_y1_r, eye_x2_r, eye_y2_r])))

    return lst_add, face_box, head_pose, keypoints, scores


def resume_from_epoch(args, model, scheduler, ckpt_dir):
    
    ckpt_load = os.path.join(ckpt_dir, 'epoch_%02d_weights.pt' % (args.resume))
    torch_load = torch.load(ckpt_load)
    state_dict = torch_load['state_dict']
    load_pretrained_weights(model, state_dict = state_dict, weight_path=ckpt_load) 
    print("successfully load {}!".format(ckpt_load))
    step = torch_load['train_step']
    if not args.no_decay: #and not args.coord_regression:
        for i in range(args.resume+1):
            scheduler.step()
    return step