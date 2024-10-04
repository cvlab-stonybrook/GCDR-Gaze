from torchvision import transforms
import numpy as np
import random
from scipy import signal
import cv2
from sklearn.metrics import roc_auc_score

# data transform for image
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}


# generate a gaussion on points in a map with im_shap
def get_paste_kernel(im_shape, points, kernel, shape=(224 // 4, 224 // 4)):
    # square kernel
    k_size = kernel.shape[0] // 2
    x, y = points
    image_height, image_width = im_shape[:2]
    x, y = int(round(image_width * x)), int(round(y * image_height))
    x1, y1 = x - k_size, y - k_size
    x2, y2 = x + k_size, y + k_size
    h, w = shape
    if x2 >= w:
        w = x2 + 1
    if y2 >= h:
        h = y2 + 1
    heatmap = np.zeros((h, w))
    left, top, k_left, k_top = x1, y1, 0, 0
    if x1 < 0:
        left = 0
        k_left = -x1
    if y1 < 0:
        top = 0
        k_top = -y1

    heatmap[top:y2+1, left:x2+1] = kernel[k_top:, k_left:]
    return heatmap[0:shape[0], 0:shape[0]]

def calc_auc(pred_hm, gazes):
    
    error_list = []
    pred = pred_hm
    pred = cv2.resize(pred, (5, 5))
    gt_points = gazes
    #pred[...] = 0.0
    #pred[2, 2] = 1.0
    gt_heatmap = np.zeros((5, 5))
    for gt_point in gt_points:
        x, y = list(map(int, list(gt_point * 5)))
        gt_heatmap[y, x] = 1.0

    auc = roc_auc_score(gt_heatmap.reshape([-1]).astype(np.int32), pred.reshape([-1]))
    return auc





def gkern(kernlen=51, std=9):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d

kernel_map = gkern(21, 3)

